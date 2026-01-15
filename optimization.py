import numpy as np
import random
import pandas as pd
import logging
import multiprocessing
import os
import math
from functools import lru_cache

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.result import Result

from network import apply_strategies
from accessibility import calculate_accessibility_optimized_single_thread
from vnd import VNDLocalSearch


# ==============================================================================
# Custom Operators (Sampling, Repair, Mutation)
# ==============================================================================

class CustomSampling(Sampling):
    """
    Heuristic Sampling based on Edge Betweenness Centrality.
    Divided edges into High, Mid, and Low priority groups to guide initial population.
    """

    def __init__(self, allowed_strategies, edges_df, config):
        super().__init__()
        self.allowed_strategies = allowed_strategies
        self.edges_df = edges_df
        self.config = config

    def _do(self, problem, n_samples, **kwargs):
        n_edges = problem.n_var
        X = np.zeros((n_samples, n_edges), dtype=int)

        # 1. Map edges to centrality
        centrality_map = {row["ID"]: row["betweenness_centrality"] for _, row in self.edges_df.iterrows()}
        edge_ids = [problem.edge_ids[i] for i in range(n_edges)]

        sortable_edges = []
        for idx, (u, v) in enumerate(edge_ids):
            if max(self.allowed_strategies[idx]) > 0:
                # Find matching row in dataframe
                match = self.edges_df[
                    ((self.edges_df["IN_ID"] == u) & (self.edges_df["OUT_ID"] == v)) |
                    ((self.edges_df["IN_ID"] == v) & (self.edges_df["OUT_ID"] == u))
                    ]
                val = centrality_map.get(match["ID"].values[0], 0.0) if not match.empty else 0.0
                sortable_edges.append((idx, val))

        # 2. Sort and partition
        sortable_edges.sort(key=lambda x: x[1], reverse=True)
        total_operable = len(sortable_edges)
        high_cut = int(0.3 * total_operable)
        mid_cut = int(0.7 * total_operable)

        high_idxs = {i for i, _ in sortable_edges[:high_cut]}
        mid_idxs = {i for i, _ in sortable_edges[high_cut:mid_cut]}

        # 3. Generate samples
        rand_ratio = self.config.get("sampling", {}).get("random_init_ratio", 0.25)
        n_random = int(rand_ratio * n_samples)

        for k in range(n_samples):
            # Pure Random Samples
            if k < n_random:
                for j, allowed in enumerate(self.allowed_strategies):
                    X[k, j] = np.random.choice(allowed)
                continue

            # Heuristic Samples
            for j in range(n_edges):
                allowed = self.allowed_strategies[j]
                if max(allowed) == 0:
                    X[k, j] = 0
                    continue

                # Assign higher probability of action to high centrality edges
                if j in high_idxs:
                    if 2 in allowed:
                        X[k, j] = np.random.choice([1, 2], p=[0.4, 0.6])
                    else:
                        X[k, j] = np.random.choice([x for x in allowed if x != 0])
                elif j in mid_idxs:
                    if 1 in allowed:
                        X[k, j] = np.random.choice([0, 1], p=[0.5, 0.5])
                    else:
                        X[k, j] = 0
                else:
                    X[k, j] = 0

        return X


class IntegerRepair(Repair):
    """
    Ensures genes (strategies) remain within valid bounds (0, 1, 2) and allowed masks.
    """

    def __init__(self, allowed_strategies):
        super().__init__()
        self.allowed_strategies = allowed_strategies

    def _do(self, problem, X, **kwargs):
        X_rounded = np.round(X).astype(int)
        for i in range(X_rounded.shape[0]):
            for j in range(X_rounded.shape[1]):
                allowed = self.allowed_strategies[j]
                val = X_rounded[i, j]
                if val not in allowed:
                    # Snap to nearest valid strategy
                    X_rounded[i, j] = min(allowed, key=lambda x: abs(x - val))
        return X_rounded


class DirectedMutation(Repair):
    """
    Custom Mutation operator guided by Betweenness Centrality.
    """

    def __init__(self, allowed_strategies, centrality_vec, config):
        super().__init__()
        self.allowed_strategies = allowed_strategies
        self.centrality_vec = list(enumerate(centrality_vec))
        self.config = config

    def _do(self, problem, X, **kwargs):
        n_edges = problem.n_var
        mut_cfg = self.config.get("mutation", {})
        replace_min = mut_cfg.get("replace_min", 0.05)
        replace_max = mut_cfg.get("replace_max", 0.15)
        directed_ratio = mut_cfg.get("directed_ratio", 0.7)

        for i in range(X.shape[0]):
            # Random Mutation
            if random.random() > directed_ratio:
                for j, allowed in enumerate(self.allowed_strategies):
                    if max(allowed) > 0 and random.random() < 0.05:
                        X[i, j] = np.random.choice(allowed)
                continue

            # Directed Mutation (Centrality Based)
            # Group current strategies
            groups = {0: [], 1: [], 2: []}
            for idx, val in enumerate(X[i]):
                groups[val].append((idx, self.centrality_vec[idx][1]))

            # Sort groups by centrality
            for key in groups:
                # Upgrade logic: prioritize high centrality. Downgrade logic: prioritize low centrality.
                groups[key].sort(key=lambda x: x[1], reverse=(key != 2))

            # Determine number of genes to mutate
            rep_count = max(1, int(random.uniform(replace_min, replace_max) * n_edges))

            # Apply logic: Upgrade high-centrality 0->1 or 1->2; Downgrade low-centrality 2->1 or 1->0
            # (Simplified logic below focuses on upgrading high value and downgrading low value to balance budget)

            # Downgrade most expensive/least critical
            for idx, _ in groups[2][:rep_count]:
                if 1 in self.allowed_strategies[idx]: X[i, idx] = 1
            for idx, _ in groups[1][:rep_count]:
                X[i, idx] = 0

            # Upgrade most critical
            for idx, _ in groups[1][-rep_count:]:
                if 2 in self.allowed_strategies[idx]: X[i, idx] = 2
            for idx, _ in groups[0][-rep_count:]:
                if 1 in self.allowed_strategies[idx]: X[i, idx] = 1

        return X


# ==============================================================================
# Callbacks
# ==============================================================================

class ChainedCallback(Callback):
    """Executes multiple callbacks in sequence."""

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def notify(self, algorithm):
        for cb in self.callbacks:
            if hasattr(cb, 'notify'): cb.notify(algorithm)

    def finalize(self):
        for cb in self.callbacks:
            if hasattr(cb, 'finalize'): cb.finalize()


class GenerationWiseCallback(Callback):
    """Saves population statistics and results at every generation."""

    def __init__(self, problem_context, output_filename="output/all_generations.csv"):
        super().__init__()
        self.ctx = problem_context
        self.output_filename = output_filename
        self.history = []

    def notify(self, algorithm):
        n_gen = algorithm.n_gen
        pop = algorithm.pop
        results_array = pop.get("results")
        if results_array is None: return

        for i in range(len(pop)):
            ind_res = results_array[i]
            # Save only valid and feasible solutions
            if not ind_res.get('valid', False) or ind_res.get('cost', float('inf')) > self.ctx['budget']:
                continue

            # Calculate normalized gains
            total_loss = (ind_res['direct_loss'] - self.ctx['optimal_direct_loss']) + ind_res['indirect_loss']
            denom_econ = self.ctx['original_loss'] - self.ctx['optimal_total_loss']
            econ_gain = 100 * (self.ctx['original_loss'] - total_loss) / denom_econ if denom_econ > 1e-9 else 0

            denom_soc = self.ctx['optimal_accessibility'] - self.ctx['A_no_action']
            soc_gain = 100 * (ind_res['A_action'] - self.ctx['A_no_action']) / denom_soc if abs(denom_soc) > 1e-9 else 0

            strategy_vec = pop[i].get("X")
            counts = np.bincount(strategy_vec, minlength=3)

            self.history.append({
                "generation": n_gen, "solution_id": i,
                "total_gain_%": ind_res['total_gain'],
                "economic_gain_%": econ_gain,
                "social_gain_%": soc_gain,
                "total_cost": ind_res['cost'],
                "direct_loss": ind_res['direct_loss'] - self.ctx['optimal_direct_loss'],
                "indirect_loss": ind_res['indirect_loss'],
                "count_0": counts[0], "count_1": counts[1], "count_2": counts[2]
            })

    def finalize(self):
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_filename, index=False)
            logging.info(f"Generation history saved to {self.output_filename}")


class GAWithVNDCallback(Callback):
    """
    Memetic Algorithm Callback:
    1. Triggers Variable Neighborhood Descent (VND) on elite solutions.
    2. Performs 'Destruction & Reconstruction' immigration to maintain diversity.
    """

    def __init__(self, problem, edges_df, config, allowed_strategies, centrality_vec, output_dir):
        super().__init__()
        self.problem = problem
        self.config = config
        self.allowed = allowed_strategies
        self.centrality_vec = centrality_vec
        self.moves_recorder = []
        self.vnd_process_filename = os.path.join(output_dir, "000_vnd_process.csv")

        # VND Settings
        vset = config.get('vnd_settings', {})
        self.vnd_enabled = vset.get('enable', False)
        self.vnd_run_every = vset.get('run_every', 5)
        self.vnd_burn_in = vset.get('burn_in', 20)
        self.vnd_topk = vset.get('topk', 5)

        # Immigration Settings
        memetic = config.get('memetic_settings', {})
        self.immigrate_interval = memetic.get('immigrate_every', 5)
        self.immigrate_ratio = memetic.get('immigrate_ratio', 0.10)

        # Pre-compute reconstruction heuristics (Cost-Benefit Ratio)
        self.reconstruction_candidates = []
        for i, (u, v) in enumerate(self.problem.edge_ids):
            if len(self.allowed[i]) > 1:
                c_m = self.problem.G_road.edges[u, v]['costs']['maintain']
                c_u = self.problem.G_road.edges[u, v]['costs']['upgrade']
                val = self.centrality_vec[i]

                if 1 in self.allowed[i] and c_m > 1e-9:
                    self.reconstruction_candidates.append({'index': i, 'strategy': 1, 'cost': c_m, 'score': val / c_m})
                if 2 in self.allowed[i] and c_u > 1e-9:
                    self.reconstruction_candidates.append({'index': i, 'strategy': 2, 'cost': c_u, 'score': val / c_u})

        self.reconstruction_candidates.sort(key=lambda x: x['score'], reverse=True)
        self.seed = config.get('random_seed', None)

    def _calculate_cost(self, vector):
        m = sum(
            self.problem.G_road.edges[u, v]['costs']['maintain'] for (u, v), s in zip(self.problem.edge_ids, vector) if
            s == 1)
        u = sum(
            self.problem.G_road.edges[u, v]['costs']['upgrade'] for (u, v), s in zip(self.problem.edge_ids, vector) if
            s == 2)
        return m + u

    def _immigrate_random(self, algorithm):
        """Replaces the worst individuals with new ones generated via destruction/reconstruction."""
        pop = algorithm.pop
        n_replace = max(1, int(self.immigrate_ratio * len(pop)))

        # Identify feasible parents
        feasible_idxs = [i for i, r in enumerate(pop.get("results")) if
                         r and r.get('valid') and r['cost'] <= self.config['budget']]
        if not feasible_idxs: return

        X_new, res_new = [], []

        for _ in range(n_replace):
            # Clone a random feasible parent
            parent_x = np.copy(pop[np.random.choice(feasible_idxs)].X)
            current_cost = self._calculate_cost(parent_x)

            # Destruction: Remove 10-20% of existing investments
            active_idxs = [i for i, s in enumerate(parent_x) if s > 0]
            if not active_idxs: continue

            n_destroy = max(1, int(random.uniform(0.1, 0.2) * len(active_idxs)))
            destroy_idxs = np.random.choice(active_idxs, size=n_destroy, replace=False)
            parent_x[destroy_idxs] = 0
            freed_budget = current_cost - self._calculate_cost(parent_x) + (self.config['budget'] - current_cost)

            # Reconstruction: Fill budget with best heuristic candidates
            for cand in self.reconstruction_candidates:
                if parent_x[cand['index']] == 0 and freed_budget >= cand['cost']:
                    parent_x[cand['index']] = cand['strategy']
                    freed_budget -= cand['cost']

            # Evaluate
            if self._calculate_cost(parent_x) <= self.config['budget']:
                res = self.problem._eval_single(tuple(parent_x))
                if res['valid']:
                    X_new.append(parent_x)
                    res_new.append(res)

        # Replace worst solutions
        if X_new:
            gains = [r['total_gain'] if r and r.get('valid') else -1e9 for r in pop.get("results")]
            worst_idxs = np.argsort(gains)[:len(X_new)]

            X_all, F_all, G_all, res_all = pop.get("X", "F", "G", "results")
            for k, idx in enumerate(worst_idxs):
                X_all[idx] = X_new[k]
                res_all[idx] = res_new[k]
                F_all[idx, 0] = -res_new[k]['total_gain']
                G_all[idx, 0] = res_new[k]['cost'] - self.config['budget']

            pop.set("X", X_all, "F", F_all, "G", G_all, "results", res_all)
            logging.info(f"[Immigration] Gen {algorithm.n_gen}: Injected {len(X_new)} new solutions.")

    def _vnd_worker(self, args):
        """Worker function for parallel VND execution."""
        idx, seed = args
        if seed:
            random.seed(seed);
            np.random.seed(seed)

        x0 = self.pop_ref[idx].X
        r0 = self.res_ref[idx]
        vnd = VNDLocalSearch(self.problem, self.config, self.centrality_vec, self.allowed)
        return idx, *vnd.run(x0, r0)

    def notify(self, algorithm):
        gen = algorithm.n_gen

        # 1. Immigration
        if gen > 0 and gen % self.immigrate_interval == 0:
            self._immigrate_random(algorithm)

        # 2. VND Logic
        if not self.vnd_enabled or gen < self.vnd_burn_in or gen % self.vnd_run_every != 0:
            return

        # Identify Elites
        pop = algorithm.pop
        feasible_w_idx = []
        for i, r in enumerate(pop.get("results")):
            if r and r.get('valid') and r['cost'] <= self.config['budget']:
                feasible_w_idx.append((r['total_gain'], i))

        if not feasible_w_idx: return

        # Sort top K
        feasible_w_idx.sort(reverse=True, key=lambda x: x[0])
        top_indices = [i for _, i in feasible_w_idx[:self.vnd_topk]]

        # Prepare for Parallel Execution
        self.pop_ref = pop  # Store reference for workers
        self.res_ref = pop.get("results")

        worker_args = [(idx, (self.seed + idx) if self.seed else None) for idx in top_indices]
        max_workers = min(len(top_indices), self.config['parallel']['max_workers'])

        # Execute VND
        if max_workers > 1:
            with multiprocessing.Pool(processes=max_workers) as pool:
                results = pool.map(self._vnd_worker, worker_args)
        else:
            results = [self._vnd_worker(args) for args in worker_args]

        # Update Population with improved solutions
        X_all, F_all, G_all, res_all = pop.get("X", "F", "G", "results")
        improved_count = 0

        for i, x_new, r_new, stats, log_str, moves in results:
            if moves:
                for m in moves: m.update({'generation': gen, 'solution_id': i})
                self.moves_recorder.extend(moves)

            old_gain = res_all[i]['total_gain']
            if r_new['total_gain'] > old_gain:
                X_all[i] = x_new
                res_all[i] = r_new
                F_all[i, 0] = -r_new['total_gain']
                G_all[i, 0] = r_new['cost'] - self.config['budget']
                improved_count += 1
                logging.info(f"  Elite {i} Improved: {old_gain:.4f} -> {r_new['total_gain']:.4f}")

        pop.set("X", X_all, "F", F_all, "G", G_all, "results", res_all)
        del self.pop_ref, self.res_ref

        if improved_count > 0:
            logging.info(f"[VND] Gen {gen}: {improved_count}/{len(top_indices)} elites improved.")

    def finalize(self):
        if self.moves_recorder:
            pd.DataFrame(self.moves_recorder).to_csv(self.vnd_process_filename, index=False)
            logging.info(f"VND trace saved to {self.vnd_process_filename}")


# ==============================================================================
# Problem Definition
# ==============================================================================

class RoadProblem(Problem):
    def __init__(self, G_road, G_composite, edge_ids, config, economic_calc,
                 A_optimal, original_loss, A_no_action, zhs_map, optimal_total_loss):

        self.edge_ids = edge_ids
        self.G_road = G_road
        self.G_composite = G_composite
        self.config = config
        self.economic_calc = economic_calc
        self.zhs_map = zhs_map

        # Baselines
        self.original_loss = original_loss
        self.optimal_total_loss = optimal_total_loss
        self.A_optimal = A_optimal
        self.A_no_action = A_no_action

        # Define allowed strategy space per edge
        self.allowed_strategies = []
        for u, v in edge_ids:
            mask = G_road.edges[u, v]['strategy_mask']  # (if_w, if_s)
            strategies = [0]
            if mask[0]: strategies.append(1)
            if mask[1]: strategies.append(2)
            self.allowed_strategies.append(strategies)

        super().__init__(
            n_var=len(edge_ids),
            n_obj=1,
            n_constr=2,
            xl=0,
            xu=2
        )

    @lru_cache(maxsize=2000)
    def _eval_single(self, strategy_tuple):
        """Evaluate a single solution vector."""
        strategy_vector = np.array(strategy_tuple, dtype=int)
        edge_id_list = [self.G_road.edges[u, v]['id'] for u, v in self.edge_ids]

        # 1. Cost
        m_costs = [self.G_road.edges[u, v]['costs']['maintain'] if s == 1 else 0
                   for (u, v), s in zip(self.edge_ids, strategy_vector)]
        u_costs = [self.G_road.edges[u, v]['costs']['upgrade'] if s == 2 else 0
                   for (u, v), s in zip(self.edge_ids, strategy_vector)]

        total_cost = sum(m_costs) + sum(u_costs)

        # 2. Network Update
        G_road_upd, _ = apply_strategies(self.G_road, strategy_vector, edge_id_list)
        G_comp_upd, _ = apply_strategies(self.G_composite, strategy_vector, edge_id_list)

        # 3. Metrics
        direct_loss = self.economic_calc.direct_loss(G_road_upd)
        indirect_loss = self.economic_calc.indirect_loss(G_road_upd)
        access_score = calculate_accessibility_optimized_single_thread(G_comp_upd, self.zhs_map)

        total_loss_action = direct_loss + indirect_loss

        # 4. Objectives (Percentages)
        denom_econ = self.original_loss - self.optimal_total_loss
        econ_obj = 100 * (self.original_loss - total_loss_action) / denom_econ if denom_econ > 1e-9 else 0

        denom_soc = self.A_optimal - self.A_no_action
        soc_obj = 100 * (access_score - self.A_no_action) / denom_soc if abs(denom_soc) > 1e-9 else 0

        total_gain = (self.config['weights']['a'] * econ_obj) + (self.config['weights']['b'] * soc_obj)

        return {
            'valid': True, 'total_gain': total_gain, 'cost': total_cost,
            'maintain_cost': sum(m_costs), 'upgrade_cost': sum(u_costs),
            'direct_loss': direct_loss, 'indirect_loss': indirect_loss, 'A_action': access_score
        }

    def _evaluate(self, X, out, *args, **kwargs):
        res, F, G = [], [], []

        for x in X:
            try:
                r = self._eval_single(tuple(x.astype(int)))
                if not r['valid']: raise ValueError

                res.append(r)
                F.append([-r['total_gain']])  # Minimize negative gain = Maximize gain
                G.append([r['cost'] - self.config['budget'], 0])
            except Exception:
                res.append({'valid': False, 'total_gain': -1e9, 'cost': 1e9})
                F.append([1e9])
                G.append([1e9, 0])

        out["F"] = np.array(F, dtype=float)
        out["G"] = np.array(G, dtype=float)
        out["results"] = res


# ==============================================================================
# Helper Algorithms
# ==============================================================================

def simulated_annealing_refine(initial_solution, problem, config):
    """Refines a single solution using Simulated Annealing."""
    sa_cfg = config.get('sa_settings', {})
    iterations = sa_cfg.get('iterations_per_elite', 100)
    temp = sa_cfg.get('initial_temp', 10.0)
    alpha = sa_cfg.get('cooling_rate', 0.995)

    curr_sol = np.copy(initial_solution)
    curr_res = problem._eval_single(tuple(curr_sol))

    best_sol = np.copy(curr_sol)
    best_res = curr_res

    operable_idxs = [i for i, allowed in enumerate(problem.allowed_strategies) if len(allowed) > 1]
    if not operable_idxs: return best_sol, best_res

    for _ in range(iterations):
        # Neighbor: mutate one gene
        neighbor = np.copy(curr_sol)
        idx = random.choice(operable_idxs)
        new_strat = random.choice([s for s in problem.allowed_strategies[idx] if s != neighbor[idx]])
        neighbor[idx] = new_strat

        n_res = problem._eval_single(tuple(neighbor))
        if not n_res['valid'] or n_res['cost'] > config['budget']: continue

        delta = n_res['total_gain'] - curr_res['total_gain']

        # Acceptance Probability
        if delta > 0 or random.random() < math.exp(delta / temp):
            curr_sol, curr_res = neighbor, n_res
            if curr_res['total_gain'] > best_res['total_gain']:
                best_sol, best_res = curr_sol, curr_res

        temp *= alpha

    return best_sol, best_res


# ==============================================================================
# Main Optimization Entry Point
# ==============================================================================

def optimize(G_road, G_composite, edge_ids, config, economic_calc,
             A_optimal, original_loss, A_no_action, edges_df, zhs_map,
             optimal_total_loss, optimal_direct_loss, output_dir):
    problem = RoadProblem(G_road, G_composite, edge_ids, config, economic_calc,
                          A_optimal, original_loss, A_no_action, zhs_map, optimal_total_loss)

    # Global Seed
    seed = config.get('random_seed', None)
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Pre-calculate Centrality Vector
    centrality_vec = []
    for u, v in edge_ids:
        row = edges_df[((edges_df["IN_ID"] == u) & (edges_df["OUT_ID"] == v)) |
                       ((edges_df["IN_ID"] == v) & (edges_df["OUT_ID"] == u))]
        val = row['betweenness_centrality'].values[0] if not row.empty else 0.0
        centrality_vec.append(val)
    centrality_vec = np.array(centrality_vec, dtype=float)

    mode = config.get('algorithm_mode', 'heuristic_ga_vnd')
    logging.info(f"===== Running Optimization Mode: {mode} =====")

    # --- Common GA Configuration ---
    pop_size = config['nsga3']['population_size']
    n_gen = config['nsga3']['generations']

    repair = IntegerRepair(problem.allowed_strategies)
    crossover = SBX(prob=config['nsga3']['crossover_prob'], eta=15, repair=repair)

    # Callbacks
    ctx = {
        'budget': config['budget'], 'original_loss': original_loss,
        'optimal_total_loss': optimal_total_loss, 'optimal_direct_loss': optimal_direct_loss,
        'original_loss': original_loss, 'A_no_action': A_no_action,
        'optimal_accessibility': A_optimal
    }
    gen_cb = GenerationWiseCallback(ctx, os.path.join(output_dir, f"000_gen_history_{mode}.csv"))
    callbacks = [gen_cb]

    # --- 1. Configure Algorithm ---
    algorithm = None

    if mode == 'standard_ga':
        algorithm = GA(
            pop_size=pop_size, sampling=IntegerRandomSampling(), crossover=crossover,
            mutation=PolynomialMutation(eta=20, repair=repair), eliminate_duplicates=True
        )

    elif mode in ['heuristic_ga', 'heuristic_ga_vnd', 'heuristic_ga_sa']:
        algorithm = GA(
            pop_size=pop_size,
            sampling=CustomSampling(problem.allowed_strategies, edges_df, config),
            crossover=crossover,
            mutation=DirectedMutation(problem.allowed_strategies, centrality_vec, config),
            eliminate_duplicates=True
        )
        if mode == 'heuristic_ga_vnd':
            vnd_cb = GAWithVNDCallback(problem, edges_df, config, problem.allowed_strategies, centrality_vec,
                                       output_dir)
            callbacks.append(vnd_cb)

    # --- 2. Execution Logic (Standard/Hybrid vs Standalone) ---

    if mode in ['standard_ga', 'heuristic_ga', 'heuristic_ga_vnd']:
        # Run Standard/VND-Hybrid GA
        res = minimize(
            problem, algorithm, ('n_gen', n_gen),
            callback=ChainedCallback(callbacks), verbose=True,
            n_jobs=min(config['parallel']['max_workers'], multiprocessing.cpu_count())
        )

    elif mode in ['heuristic_ga_sa', 'standard_ga_sa']:
        # Run GA then refine elites with SA
        res = minimize(
            problem, algorithm, ('n_gen', n_gen),
            callback=ChainedCallback(callbacks), verbose=True,
            n_jobs=min(config['parallel']['max_workers'], multiprocessing.cpu_count())
        )
        logging.info("--- Phase 2: Refine Elites with Simulated Annealing ---")

        pop = res.pop
        feas = [i for i in pop if i.G[0] <= 0]
        feas.sort(key=lambda x: x.F[0])

        elites = feas[:config['sa_settings'].get('num_elites_for_sa', 5)]
        best_overall = elites[0] if elites else None

        for i, ind in enumerate(elites):
            logging.info(f"Refining Elite {i + 1}...")
            x_new, r_new = simulated_annealing_refine(ind.X, problem, config)
            if r_new['total_gain'] > -best_overall.F[0]:
                best_overall.X = x_new
                best_overall.F[0] = -r_new['total_gain']
                best_overall.G[0] = r_new['cost'] - config['budget']
                best_overall.set("results", r_new)

        res.pop = Population.create(best_overall)
        res.opt = res.pop

    elif mode in ['vnd_only', 'sa_only']:
        # Standalone Heuristic Search (Multiple Restarts)
        logging.info(f"--- Running Standalone {mode} ---")
        n_starts = config['standalone_settings'].get('num_initial_solutions', 10)
        sampler = CustomSampling(problem.allowed_strategies, edges_df, config)

        best_res_global = {'total_gain': -1e9}
        best_x_global = None

        # Override eval to track progress
        orig_eval = problem._eval_single
        log = []

        def tracked_eval(x):
            r = orig_eval(x)
            if r['valid']: log.append(r['total_gain'])
            return r

        problem._eval_single = tracked_eval

        for i in range(n_starts):
            # Find feasible start
            x0 = None
            for _ in range(200):
                cand = sampler._do(problem, 1)[0]
                if problem._eval_single(tuple(cand))['cost'] <= config['budget']:
                    x0 = cand;
                    break
            if x0 is None: x0 = np.zeros(problem.n_var, dtype=int)

            r0 = problem._eval_single(tuple(x0))

            # Search
            if mode == 'vnd_only':
                vnd = VNDLocalSearch(problem, config, centrality_vec, problem.allowed_strategies)
                x_final, r_final, _, _, _ = vnd.run(x0, r0)
            else:
                x_final, r_final = simulated_annealing_refine(x0, problem, config)

            if r_final['total_gain'] > best_res_global['total_gain']:
                best_res_global = r_final
                best_x_global = x_final
                logging.info(f"New Best found at start {i + 1}: {r_final['total_gain']:.4f}")

        # Construct Result
        res = Result()
        opt = Population.new(X=[best_x_global])
        opt.set("F", [[-best_res_global['total_gain']]])
        opt.set("G", [[best_res_global['cost'] - config['budget'], 0]])
        opt.set("results", [best_res_global])
        res.opt = opt
        res.pop = opt

        # Save simple log
        pd.DataFrame(log, columns=['gain']).to_csv(os.path.join(output_dir, f"000_standalone_{mode}_log.csv"))

    else:
        raise ValueError(f"Unknown algorithm mode: {mode}")

    ChainedCallback(callbacks).finalize()
    return res