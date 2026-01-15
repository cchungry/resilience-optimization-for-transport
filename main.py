import pandas as pd
from tqdm import tqdm
import time
import logging
import json
import ast
import numpy as np
import os
from multiprocessing import cpu_count
from pymoo.config import Config

# Suppress pymoo compilation warnings
Config.warnings['not_compiled'] = False

from config import load_config
from network import build_base_network, apply_strategies
from economic import EconomicCalculator, _network_efficiency_ig_exact
from accessibility import calculate_accessibility_optimized_parallel, get_accessibility_details
from optimization import optimize, RoadProblem


def validate_floodinfo(edges_df, floodinfo_column):
    """
    Validates and standardizes the JSON format in the flood info column.
    """
    logging.info(f"Validating format for column: {floodinfo_column}...")

    for idx, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Validating Floodinfo"):
        floodinfo = row[floodinfo_column]
        if pd.notna(floodinfo):
            # Parse string to JSON/Dict
            if isinstance(floodinfo, str):
                try:
                    json.loads(floodinfo)
                except json.JSONDecodeError:
                    try:
                        # Attempt to parse non-standard JSON (e.g., single quotes)
                        parsed = ast.literal_eval(floodinfo)
                        if isinstance(parsed, dict):
                            edges_df.at[idx, floodinfo_column] = json.dumps({str(k): v for k, v in parsed.items()})
                        else:
                            raise ValueError
                    except (ValueError, SyntaxError) as e:
                        logging.error(f"Format error in Road {row['ID']}: {floodinfo}, Error: {e}")
                        raise
            elif not isinstance(floodinfo, dict):
                logging.error(f"Invalid type for Road {row['ID']}: {type(floodinfo)}")
                raise
    return edges_df


def main():
    """
    Main execution entry point for the Multi-Objective Road Network Optimization.
    """
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- File Paths ---
    ROADS_FILE_PATH = ""
    IO_TABLE_PATH = ""
    ZHS_FILE_PATH = ""
    CONFIG_FILE_PATH = ""
    OUTPUT_DIR = ""  # Changed to relative path for public release

    # --- Scenario Configuration ---
    DEPTH_COLUMN = ""
    FLOODINFO_COLUMN = ""

    RUN_SANITY_CHECK = True

    try:
        # 1. Load Configuration
        logging.info(f"Loading configuration from: {CONFIG_FILE_PATH}...")
        config = load_config(CONFIG_FILE_PATH)
        num_workers = min(config['parallel']['max_workers'], cpu_count())
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 2. Load Data
        logging.info("Loading datasets...")
        edges_df = pd.read_excel(ROADS_FILE_PATH)
        io_df = pd.read_excel(IO_TABLE_PATH, index_col=0)

        try:
            zhs_df = pd.read_excel(ZHS_FILE_PATH)
            zhs_map = pd.Series(zhs_df.ZHS.values, index=zhs_df.ID.astype(str)).to_dict()
            logging.info("ZHS data loaded successfully.")
        except FileNotFoundError:
            logging.error(f"ZHS file not found: {ZHS_FILE_PATH}")
            zhs_map = {}

        # 3. Validation
        required_columns = ['ID', 'IN_ID', 'OUT_ID', 'Length', 'original_time',
                            DEPTH_COLUMN, FLOODINFO_COLUMN, 'maintainance_cost',
                            'updated_cost', 'construction_cost', 'if_w', 'if_s', 'speed(m/min)',
                            'betweenness_centrality']

        missing_cols = set(required_columns) - set(edges_df.columns)
        if missing_cols:
            logging.error(f"Missing columns in roads.xlsx: {missing_cols}")
            raise KeyError("Missing required columns.")

        edges_df = validate_floodinfo(edges_df, FLOODINFO_COLUMN)

        # Check operability (at least one edge must be maintainable or upgradeable)
        e_edges = edges_df[edges_df['ID'].str.startswith('E')]
        n_w = sum(e_edges['if_w'])
        n_s = sum(e_edges['if_s'])
        logging.info(f"Total Edges: {len(e_edges)} | Maintainable: {n_w} | Upgradeable: {n_s}")
        if n_w == 0 and n_s == 0:
            raise ValueError("No edges available for maintenance or upgrade.")

        # 4. Build Network and Initialize Calculator
        logging.info("Building network topology...")
        G_road, G_composite = build_base_network(edges_df, DEPTH_COLUMN, FLOODINFO_COLUMN)
        economic_calc = EconomicCalculator(io_df, config['unit_cost'])

        # 5. Pre-compute Baselines
        logging.info("Calculating baseline metrics (Optimal, Original, No Action)...")

        # 5a. Construct 'Optimal Scenario' (No Floods)
        operable_edge_ids = {
            data['id'] for u, v, data in G_road.edges(data=True)
            if data['id'].startswith('E') and (data['strategy_mask'][0] or data['strategy_mask'][1])
        }
        G_road_optimal = G_road.copy()
        G_composite_optimal = G_composite.copy()

        for G_opt in [G_road_optimal, G_composite_optimal]:
            for u, v, data in G_opt.edges(data=True):
                if data['id'] in operable_edge_ids:
                    data['Floodinfo'] = {}
                    data['current_depth'] = 0.0
                    data['current_time'] = data['original_time']

        # 5b. Calculate Efficiency Baselines
        eff_optimal = _network_efficiency_ig_exact(G_road_optimal)
        eff_original = _network_efficiency_ig_exact(G_road)
        economic_calc.set_baseline_efficiencies(eff_optimal, eff_original)

        # 5c. Calculate Loss and Accessibility Baselines
        original_direct_loss = economic_calc.direct_loss(G_road)
        original_indirect_loss = economic_calc.indirect_loss(G_road)
        original_loss = original_direct_loss + original_indirect_loss

        optimal_direct_loss = economic_calc.direct_loss(G_road_optimal)
        optimal_indirect_loss = economic_calc.indirect_loss(G_road_optimal)
        optimal_total_loss = optimal_direct_loss + optimal_indirect_loss

        A_no_action = calculate_accessibility_optimized_parallel(G_composite, zhs_map, num_workers)
        optimal_accessibility = calculate_accessibility_optimized_parallel(G_composite_optimal, zhs_map, num_workers)

        logging.info(
            f"Baselines :: OptLoss: {optimal_total_loss:.2f} | OptAccess: {optimal_accessibility:.4f} | OptEff: {eff_optimal:.6f}")

        # 6. Optimization Preparation
        # Filter edges that are eligible for strategy application
        edge_ids = [
            (u, v) for u, v, data in G_road.edges(data=True)
            if data['id'].startswith('E') and (data['strategy_mask'][0] or data['strategy_mask'][1])
        ]
        logging.info(f"Optimizable edges count: {len(edge_ids)}")

        # 7. Sanity Check (Optional Debugging)
        if RUN_SANITY_CHECK:
            logging.info("===== Starting Sanity Check (Extreme Strategies) =====")
            temp_problem = RoadProblem(G_road, G_composite, edge_ids, config, economic_calc,
                                       optimal_accessibility, original_loss, A_no_action, zhs_map, optimal_total_loss)

            # Strategies: All Zeros (Do Nothing) vs All Max Possible (All In)
            do_nothing_vector = np.zeros(len(edge_ids), dtype=int)
            all_in_vector = np.array([max(allowed) if allowed else 0 for allowed in temp_problem.allowed_strategies])

            eval_nothing = temp_problem._eval_single(tuple(do_nothing_vector))
            eval_all_in = temp_problem._eval_single(tuple(all_in_vector))

            print("\n" + "=" * 60)
            print("SANITY CHECK RESULTS")
            print("=" * 60)
            print(f"{'Metric':<25} | {'Do Nothing':<15} | {'All-In Strategy':<15}")
            print("-" * 60)
            print(f"{'Total Gain':<25} | {eval_nothing['total_gain']:<15.4f} | {eval_all_in['total_gain']:<15.4f}")
            print(f"{'Total Cost':<25} | {eval_nothing['cost']:<15.2f} | {eval_all_in['cost']:<15.2f}")
            print(
                f"{'Accessibility (A_action)':<25} | {eval_nothing['A_action']:<15.4f} | {eval_all_in['A_action']:<15.4f}")
            print("=" * 60 + "\n")

        # 8. Run Optimization
        logging.info("Starting Optimization Process...")
        res = optimize(G_road, G_composite, edge_ids, config, economic_calc,
                       optimal_accessibility, original_loss, A_no_action,
                       edges_df, zhs_map, optimal_total_loss, optimal_direct_loss, output_dir=OUTPUT_DIR)

        # 9. Process Results
        logging.info("Processing optimization results...")
        final_population = res.pop

        if final_population is None or len(final_population) == 0:
            raise ValueError("Optimization Failed: Empty population.")

        # Filter feasible solutions (valid and within budget)
        feasible_solutions = []
        for ind in final_population:
            r = ind.get("results")
            if r and r.get('valid', False) and r.get('cost', float('inf')) <= config['budget']:
                feasible_solutions.append({'solution_vector': ind.X, 'results_dict': r})

        if not feasible_solutions:
            raise ValueError("Optimization Failed: No feasible solutions found within budget.")

        # Sort by Total Gain descending
        feasible_solutions.sort(key=lambda s: s['results_dict']['total_gain'], reverse=True)
        best_solution = feasible_solutions[0]
        logging.info(f"Best solution found (Gain: {best_solution['results_dict']['total_gain']:.4f}).")

        # 10. Save Output
        results_to_save = []
        x = best_solution['solution_vector']
        r = best_solution['results_dict']

        edge_id_list = [G_road.edges[u, v]['id'] for u, v in edge_ids]
        counts = np.bincount(x, minlength=3)

        # Recalculate gains/losses for reporting
        absolute_direct_loss = r.get('direct_loss', 0.0)
        direct_loss_for_report = absolute_direct_loss - optimal_direct_loss
        total_loss_after_action = direct_loss_for_report + r['indirect_loss']

        # Normalization denominators
        denom_econ = original_loss - optimal_total_loss
        denom_social = optimal_accessibility - A_no_action

        econ_gain_pct = 100 * (original_loss - (
                    absolute_direct_loss + r['indirect_loss'])) / denom_econ if denom_econ > 1e-9 else 0
        social_gain_pct = 100 * (r['A_action'] - A_no_action) / denom_social if denom_social > 1e-9 else 0

        results_to_save.append({
            "solution_id": 0,
            "strategy_mapping": str(dict(zip(edge_id_list, x.tolist()))),
            "economic_gain_%": econ_gain_pct,
            "social_gain_%": social_gain_pct,
            "total_gain_%": r['total_gain'],
            "total_cost": r['cost'],
            "maintain_cost": r.get('maintain_cost', 0.0),
            "upgrade_cost": r.get('upgrade_cost', 0.0),
            "remain_budget": config['budget'] - r['cost'],
            "direct_loss": direct_loss_for_report,
            "indirect_loss": r['indirect_loss'],
            "total_loss": total_loss_after_action,
            "A_action": r['A_action'],
            "original_direct_loss": original_direct_loss,
            "original_total_loss": original_loss,
            "A_no_action": A_no_action,
            "optimal_total_loss": optimal_total_loss,
            "A_optimal": optimal_accessibility,
            "count_strategy_0": counts[0],
            "count_strategy_1": counts[1],
            "count_strategy_2": counts[2]
        })

        # Save Pareto Front / Best Solution
        pd.DataFrame(results_to_save).to_csv(os.path.join(OUTPUT_DIR, "000_pareto_front.csv"), index=False)
        logging.info("Saved: pareto_front.csv")

        # Apply best strategies to network to generate detailed reports
        edge_id_list_full = [data['id'] for _, _, data in G_road.edges(data=True)]
        edge_ids_operable = [G_road.edges[u, v]['id'] for u, v in edge_ids]

        full_strategy_vector = np.zeros(len(edge_id_list_full), dtype=int)
        mapping_operable = dict(zip(edge_ids_operable, x))
        for i, eid in enumerate(edge_id_list_full):
            if eid in mapping_operable:
                full_strategy_vector[i] = mapping_operable[eid]

        G_road_final, _ = apply_strategies(G_road, full_strategy_vector, edge_id_list_full)
        G_composite_final, _ = apply_strategies(G_composite, full_strategy_vector, edge_id_list_full)

        # Save detailed IO Loss
        io_loss_details, _ = economic_calc.get_indirect_loss_details(G_road_final)
        pd.DataFrame([io_loss_details]).to_csv(os.path.join(OUTPUT_DIR, "000_io_loss.csv"), index=False)
        logging.info("Saved: io_loss.csv")

        # Save detailed Accessibility
        accessibility_details, _ = get_accessibility_details(G_composite_final, zhs_map)
        pd.DataFrame(accessibility_details).to_csv(os.path.join(OUTPUT_DIR, "000_A_nodes_o.csv"), index=False)
        logging.info("Saved: A_nodes_o.csv")

        elapsed_time = time.time() - start_time
        logging.info(f"Program completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        logging.error(f"Program Execution Failed: {e}")
        raise


if __name__ == "__main__":
    main()