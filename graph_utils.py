import networkx as nx
import igraph as ig


def convert_nx_to_ig(G_nx: nx.Graph):
    """
    Converts a NetworkX graph object to an igraph graph object.

    This conversion is necessary to leverage igraph's highly optimized C-core
    for performance-intensive tasks like calculating shortest paths or
    global network efficiency.

    Args:
        G_nx (nx.Graph): The source NetworkX graph.

    Returns:
        tuple: (ig_graph, nx_to_ig_map, ig_to_nx_map)
            - ig_graph (ig.Graph): The converted igraph object.
            - nx_to_ig_map (dict): Mapping from NetworkX node names to igraph integer IDs.
            - ig_to_nx_map (list): Mapping from igraph integer IDs to NetworkX node names.
    """
    # igraph requires integer vertex IDs starting from 0.
    # We create a mapping to preserve original node labels.
    node_names = list(G_nx.nodes())
    nx_to_ig_map = {name: i for i, name in enumerate(node_names)}

    # Initialize an empty undirected igraph
    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(len(node_names))

    # Convert edges to integer pairs based on the mapping
    edges = [(nx_to_ig_map[u], nx_to_ig_map[v]) for u, v in G_nx.edges()]
    ig_graph.add_edges(edges)

    # Transfer edge weights
    # Priority: 'current_time' > 'weight' > default 1.0
    weights = [
        data.get('current_time', data.get('weight', 1.0))
        for u, v, data in G_nx.edges(data=True)
    ]
    ig_graph.es['weight'] = weights

    return ig_graph, nx_to_ig_map, node_names