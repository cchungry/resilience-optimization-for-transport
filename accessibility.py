import logging
import networkx as nx
import igraph as ig
from graph_utils import convert_nx_to_ig

CUTOFF_TIME = "" #you can set your own threshold


def get_accessibility_details(G_nx, zhs_map):
    """
    Calculates detailed accessibility metrics for Origin (O) nodes and the aggregated score using igraph.

    Args:
        G_nx (nx.Graph): The networkx graph object (composite network).
        zhs_map (dict): Mapping of O-node IDs to their importance weights (ZHS).

    Returns:
        tuple: (details_list, final_score)
               - details_list: List of dicts containing per-node accessibility info.
               - final_score: The weighted global accessibility score (0.0 to 1.0).
    """
    # Identify O (Origin) and D (Destination) nodes
    o_nodes_nx = [n for n in G_nx.nodes() if n.startswith('O')]
    d_nodes_nx = {n for n in G_nx.nodes() if n.startswith('D')}

    if not o_nodes_nx or not d_nodes_nx:
        return [], 0.0

    # Convert to igraph for high-performance shortest path calculation
    G_ig, nx_to_ig, ig_to_nx = convert_nx_to_ig(G_nx)

    # Get igraph indices for O-nodes
    o_nodes_ig = [nx_to_ig[name] for name in o_nodes_nx if name in nx_to_ig]

    # Calculate shortest paths from all O-nodes to all other nodes in one go
    # weights='weight' corresponds to 'current_time' mapped in graph_utils
    all_paths = G_ig.shortest_paths(source=o_nodes_ig, weights='weight')

    details_list = []
    actual_weighted_reachability = 0.0

    for i, o_node_ig_id in enumerate(o_nodes_ig):
        o_node_name = ig_to_nx[o_node_ig_id]
        zhs_value = zhs_map.get(o_node_name, 0)

        # Retrieve pre-calculated paths for this O-node
        path_lengths_from_o = all_paths[i]
        reachable_d_count = 0

        # Check connectivity to every D-node within the cutoff time
        for d_node_name in d_nodes_nx:
            if d_node_name in nx_to_ig:
                d_node_ig_id = nx_to_ig[d_node_name]
                if path_lengths_from_o[d_node_ig_id] <= CUTOFF_TIME:
                    reachable_d_count += 1

        # Store details
        details_list.append({
            'o_node_id': o_node_name,
            'zhs_value': zhs_value,
            'reachable_d_count': reachable_d_count
        })

        actual_weighted_reachability += reachable_d_count * zhs_value

    # Normalize the score
    total_possible_weighted_reachability = len(d_nodes_nx) * sum(zhs_map.get(o, 0) for o in o_nodes_nx)

    if total_possible_weighted_reachability > 1e-9:
        final_score = actual_weighted_reachability / total_possible_weighted_reachability
    else:
        final_score = 0.0

    return details_list, final_score


def calculate_accessibility_optimized_single_thread(G, zhs_map):
    """
    Wrapper for single-threaded accessibility calculation.
    """
    _, final_score = get_accessibility_details(G, zhs_map)
    return final_score


def calculate_accessibility_optimized_parallel(G, zhs_map, num_workers):
    """
    Wrapper for accessibility calculation (interface compatibility).

    Note: Since igraph's C-core implementation is highly optimized,
    explicit Python-level multiprocessing is often unnecessary or
    slower due to overhead for this specific subgraph scale.
    """
    _, final_score = get_accessibility_details(G, zhs_map)
    return final_score


def precompute_no_flood_accessibility(edges_df, zhs_map, num_workers):
    """
    Constructs a temporary 'No Flood' graph to compute the baseline accessibility.
    """
    logging.info("Building temporary network for 'No Flood' baseline accessibility...")
    G_temp = nx.Graph()
    for _, row in edges_df.iterrows():
        # Use 'original_time' as the weight for the baseline scenario
        G_temp.add_edge(str(row['IN_ID']), str(row['OUT_ID']), current_time=row['original_time'])

    return calculate_accessibility_optimized_parallel(G_temp, zhs_map, num_workers)