import numpy as np
import logging
import random
from dataclasses import dataclass


@dataclass
class VNDSettings:
    """
    Configuration settings for the Variable Neighborhood Descent (VND) algorithm.
    """
    topk_for_each_gen: int = 5  # Number of elite solutions to refine per generation
    burn_in_gens: int = 20  # Generations to wait before enabling VND
    vnd_every: int = 2  # Frequency of VND execution (every N generations)
    step_limit: int = 100  # Max steps for the main VND loop
    time_limit_sec: float = 60  # Time limit per solution (if implemented)


class VNDLocalSearch:
    """
    Implements Variable Neighborhood Descent (VND) to refine solutions locally.

    Neighborhood Structures:
    - N1: Swap Strategy 0 (No Action) <-> Strategy 1 (Maintenance)
    - N2: Swap Strategy 1 (Maintenance) <-> Strategy 2 (Upgrade)
    - N3: Swap Strategy 0 (No Action) <-> Strategy 2 (Upgrade)

    Heuristic:
    Uses 'Betweenness Centrality' to guide the search towards critical edges
    rather than purely random swaps.
    """

    def __init__(self, problem, config, centrality_vec, allowed_strategies):
        self.problem = problem
        self.config = config
        self.centrality = np.asarray(centrality_vec)
        self.allowed = allowed_strategies
        self.budget = config['budget']

        vset = self.config.get('vnd_settings', {})
        self.swap_limit = vset.get('swap_limit_per_n', 50)

    def _calculate_cost(self, solution_vector):
        """
        Calculates the financial cost of a solution vector efficiently.
        """
        m_cost = sum(self.problem.G_road.edges[u, v]['costs']['maintain'] if s == 1 else 0
                     for (u, v), s in zip(self.problem.edge_ids, solution_vector))
        u_cost = sum(self.problem.G_road.edges[u, v]['costs']['upgrade'] if s == 2 else 0
                     for (u, v), s in zip(self.problem.edge_ids, solution_vector))
        return m_cost + u_cost

    def _better(self, new_result, current_result):
        """Checks if the new result is strictly better than the current one."""
        return (new_result and
                new_result.get('valid', False) and
                new_result['total_gain'] > current_result['total_gain'])

    def _explore_neighborhood(self, x, r_cur, s_low, s_high, stats, neighborhood_name):
        """
        Explores a specific neighborhood defined by swapping two strategies (s_low <-> s_high).

        Optimization Logic:
        1. Identify candidates: Edges currently assigned s_low (but allow s_high) and vice versa.
        2. Heuristic Sort: Sort candidates by Betweenness Centrality.
        3. Subset Selection: Focus on the top 50% most central edges to reduce search space.
        4. Random Swaps: Perform random swaps within this high-priority subset.
        """
        x_best, r_best = np.copy(x), r_cur.copy()
        was_improved = False
        moves = []

        # 1. Filter candidates based on current state and allowed strategies
        indices_low = [i for i, s in enumerate(x_best) if s == s_low and s_high in self.allowed[i]]
        indices_high = [i for i, s in enumerate(x_best) if s == s_high and s_low in self.allowed[i]]

        # 2. Sort by Centrality (High to Low for importance)
        # Note: Strategy 'High' usually benefits high centrality edges more,
        # so we might want to prioritize upgrading central edges.
        sorted_low = sorted(indices_low, key=lambda i: self.centrality[i], reverse=True)
        sorted_high = sorted(indices_high, key=lambda i: self.centrality[i], reverse=False)

        # 3. Define Core Subset (Top 50%)
        min_size = min(len(sorted_low), len(sorted_high))
        subset_size = int(min_size * 0.5)

        if subset_size == 0:
            return x_best, r_best, was_improved, moves

        subset_low = sorted_low[:subset_size]
        subset_high = sorted_high[:subset_size]

        # 4. Smart Stop Condition
        max_possible_swaps = len(subset_low) * len(subset_high)
        tried_pairs = set()

        # 5. Search Loop
        for swap_count in range(self.swap_limit):
            if len(tried_pairs) >= max_possible_swaps:
                logging.debug(f"[VND] {neighborhood_name}: Exhausted all {max_possible_swaps} combinations.")
                break

            stats[f'{neighborhood_name}_total_attempts'] += 1

            # a. Pick a random pair from the subset
            while True:
                idx_low = random.choice(subset_low)
                idx_high = random.choice(subset_high)
                pair_key = tuple(sorted((idx_low, idx_high)))
                if pair_key not in tried_pairs:
                    tried_pairs.add(pair_key)
                    break

            # b. Generate Neighbor (Swap strategies)
            x_neighbor = np.copy(x_best)
            x_neighbor[idx_low], x_neighbor[idx_high] = x_neighbor[idx_high], x_neighbor[idx_low]

            # c. Budget Check
            cost_neighbor = self._calculate_cost(x_neighbor)
            if cost_neighbor > self.budget:
                stats[f'{neighborhood_name}_over_budget_attempts'] += 1
                continue

            # d. Evaluate
            stats[f'{neighborhood_name}_feasible_attempts'] += 1
            r_neighbor = self.problem._eval_single(tuple(x_neighbor))

            # e. Acceptance Criteria
            if self._better(r_neighbor, r_best):
                stats[f'{neighborhood_name}_improvements_found'] += 1

                # Log the move
                moves.append({
                    'move_type': neighborhood_name,
                    'index': f"{idx_low},{idx_high}",
                    'old_val': f"{x_best[idx_low]},{x_best[idx_high]}",
                    'new_val': f"{x_neighbor[idx_low]},{x_neighbor[idx_high]}",
                    'old_total_gain': r_best['total_gain'],
                    'new_total_gain': r_neighbor['total_gain'],
                    'old_cost': r_best['cost'],
                    'new_cost': r_neighbor['cost']
                })

                # Accept new solution
                x_best, r_best = x_neighbor, r_neighbor
                was_improved = True

                # Dynamic Update: Re-evaluate subsets after a successful swap
                # This ensures we are always swapping based on the NEW configuration
                indices_low = [i for i, s in enumerate(x_best) if s == s_low and s_high in self.allowed[i]]
                indices_high = [i for i, s in enumerate(x_best) if s == s_high and s_low in self.allowed[i]]

                sorted_low = sorted(indices_low, key=lambda i: self.centrality[i], reverse=True)
                sorted_high = sorted(indices_high, key=lambda i: self.centrality[i], reverse=False)

                min_size = min(len(sorted_low), len(sorted_high))
                subset_size = int(min_size * 0.5)

                if subset_size == 0:
                    break

                subset_low = sorted_low[:subset_size]
                subset_high = sorted_high[:subset_size]

                tried_pairs.clear()
                max_possible_swaps = len(subset_low) * len(subset_high)
            else:
                stats[f'{neighborhood_name}_non_improvements'] += 1

        return x_best, r_best, was_improved, moves

    def run(self, x0, r0):
        """
        Executes the VND procedure starting from solution x0.
        Sequence: N1 -> N2 -> N3
        """
        x_best, r_best = np.copy(x0), r0.copy()

        # Initialize statistics
        run_stats = {f'N{i}_{key}': 0 for i in [1, 2, 3] for key in
                     ['total_attempts', 'over_budget_attempts', 'feasible_attempts', 'improvements_found',
                      'non_improvements']}
        all_moves = []

        # Iterate through neighborhoods
        # k=1: N1 (0 <-> 1)
        # k=2: N2 (1 <-> 2)
        # k=3: N3 (0 <-> 2)
        k = 1
        while k <= 3:
            neighborhood_name = f"N{k}"

            if k == 1:
                x_best, r_best, _, moves = self._explore_neighborhood(x_best, r_best, 0, 1, run_stats,
                                                                      neighborhood_name)
            elif k == 2:
                x_best, r_best, _, moves = self._explore_neighborhood(x_best, r_best, 1, 2, run_stats,
                                                                      neighborhood_name)
            elif k == 3:
                x_best, r_best, _, moves = self._explore_neighborhood(x_best, r_best, 0, 2, run_stats,
                                                                      neighborhood_name)

            all_moves.extend(moves)
            k += 1

        total_improvements = sum(run_stats[f'N{i}_improvements_found'] for i in [1, 2, 3])

        log_string = (
            f"VND Status: {'Improved' if total_improvements > 0 else 'No Change'}. "
            f"Stats (Attempts/Success): "
            f"N1=({run_stats['N1_total_attempts']}/{run_stats['N1_improvements_found']}); "
            f"N2=({run_stats['N2_total_attempts']}/{run_stats['N2_improvements_found']}); "
            f"N3=({run_stats['N3_total_attempts']}/{run_stats['N3_improvements_found']})"
        )

        return x_best, r_best, run_stats, log_string, all_moves