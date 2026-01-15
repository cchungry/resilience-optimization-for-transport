import json
import ast
import networkx as nx
import numpy as np
import pandas as pd
import logging


def _calculate_travel_time(length, base_speed, depth):
    """
    Calculates travel time based on length, base speed, and flood depth.
    Uses an exponential decay function for speed reduction: v = v0 * e^(-9 * h)
    """
    if base_speed <= 0:
        return float('inf')

    # Speed decay formula
    reduced_speed = base_speed * np.exp(-9 * depth)

    if reduced_speed < 1e-6:
        return float('inf')

    return length / reduced_speed


def build_base_network(edges_df, depth_column, floodinfo_column):
    """
    Constructs the base road network (G_road) and composite network (G_composite).

    Args:
        edges_df (pd.DataFrame): DataFrame containing edge data.
        depth_column (str): Column name for aggregated flood depth.
        floodinfo_column (str): Column name for segmented flood info (JSON).

    Returns:
        tuple: (G_road, G_composite)
            - G_road: Graph containing only road edges (starting with 'E').
            - G_composite: Graph containing all edges (Roads + Origin/Destination links).
    """
    G_road = nx.Graph()
    G_composite = nx.Graph()

    for _, row in edges_df.iterrows():
        # --- 1. Parse Floodinfo ---
        floodinfo = row[floodinfo_column]
        if pd.notna(floodinfo):
            if isinstance(floodinfo, str):
                try:
                    floodinfo = json.loads(floodinfo)
                except json.JSONDecodeError:
                    try:
                        # Fallback for non-standard JSON (e.g., single quotes)
                        floodinfo = ast.literal_eval(floodinfo)
                    except (ValueError, SyntaxError):
                        logging.error(f"Failed to parse Floodinfo for Road {row['ID']}")
                        raise ValueError(f"Invalid Floodinfo format for {row['ID']}")
            elif isinstance(floodinfo, dict):
                pass
            else:
                raise ValueError(f"Invalid Floodinfo type for Road {row['ID']}")
        else:
            # Default: If no flood info, assume entire length is at depth 0
            floodinfo = {"0.0": row['Length']}

        # Ensure keys are floats
        try:
            floodinfo = {float(k): float(v) for k, v in floodinfo.items()}
        except Exception as e:
            logging.error(f"Error converting Floodinfo keys/values for Road {row['ID']}: {e}")
            raise

        # --- 2. Build Edge Attributes ---
        edge_data = {
            'id': row['ID'],
            'length': row['Length'],
            'original_time': row['original_time'],
            'current_time': row['original_time'],
            'Floodinfo': floodinfo,
            'strategy_mask': (row['if_w'], row['if_s']),  # (Allow Maintenance, Allow Upgrade)
            'costs': {
                'maintain': row['maintainance_cost'] if pd.notna(row['maintainance_cost']) else 0.0,
                'upgrade': row['updated_cost'] if pd.notna(row['updated_cost']) else 0.0
            },
            'construction_cost': row['construction_cost'] if pd.notna(row['construction_cost']) else 0.0
        }

        # --- 3. Compute Initial State for Roads ---
        if row['ID'].startswith('E'):
            edge_data['current_depth'] = row[depth_column]
            edge_data['base_speed'] = row['speed(m/min)'] if pd.notna(row['speed(m/min)']) else 0.0

            if edge_data['base_speed'] == 0:
                logging.error(f"Road {row['ID']} has 0 speed.")
                raise ValueError("Invalid speed.")

            edge_data['current_time'] = _calculate_travel_time(
                edge_data['length'],
                edge_data['base_speed'],
                edge_data['current_depth']
            )

        # --- 4. Add to Graphs ---
        G_composite.add_edge(row['IN_ID'], row['OUT_ID'], **edge_data)
        if row['ID'].startswith('E'):
            G_road.add_edge(row['IN_ID'], row['OUT_ID'], **edge_data)

    logging.info(f"Network Built :: Nodes: {G_road.number_of_nodes()} | Edges: {G_road.number_of_edges()}")
    return G_road, G_composite


def apply_strategies(G, strategy_vector, edge_ids):

    G_updated = G.copy()
    strategies = dict(zip(edge_ids, strategy_vector))
    floodinfo_changes = {}

    for u, v, data in G_updated.edges(data=True):
        edge_id = data['id']
        strategy = strategies.get(edge_id, 0)
        if_w, if_s = data['strategy_mask']

        # Validation: Ensure strategy is allowed for this edge
        if (strategy == 1 and not if_w) or (strategy == 2 and not if_s):
            logging.error(f"Illegal strategy {strategy} applied to edge {edge_id}")
            return None, {}

        # --- Strategy 2: Upgrade (Complete Restoration) ---
        if strategy == 2:
            data['Floodinfo'] = {}  # Depth becomes 0
            data['current_depth'] = 0.0
            data['current_time'] = data['original_time']

        # --- Strategy 1: Maintenance (Mitigation) ---
        elif strategy == 1:
            original_floodinfo = data.get('Floodinfo', {})

            # 1. Reduce flood depth by 80% (retain 20%)
            # This mainly affects direct economic loss calculation logic in economic.py
            data['Floodinfo'] = {float(d) * "": l for d, l in original_floodinfo.items()}  # set efficiency

            # 2. Update functional travel time
            if not data['Floodinfo']:
                data['current_depth'] = 0.0
            else:
                # Recalculate weighted average depth
                total_length = sum(data['Floodinfo'].values())
                if total_length > 0:
                    weighted_depth = sum(d * l for d, l in data['Floodinfo'].items()) / total_length
                    data['current_depth'] = weighted_depth
                else:
                    data['current_depth'] = 0.0

            # 3. Recalculate time with upper bound cap
            v0 = data.get('base_speed')
            if v0:
                calculated_time = _calculate_travel_time(data['length'], v0, data['current_depth'])
                # Constraint: Time cannot exceed 3x the original design time
                data['current_time'] = min(calculated_time, data['original_time'] * 3)

    return G_updated, floodinfo_changes