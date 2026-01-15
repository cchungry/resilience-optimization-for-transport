import numpy as np
import logging
import igraph as ig
from graph_utils import convert_nx_to_ig


def _network_efficiency_ig_exact(G_nx):
    """
    Calculates the exact Global Network Efficiency using igraph.
    Formula: E = 1/(N(N-1)) * Sum(1/d_ij)

    This function correctly handles disconnected graphs by summing efficiencies 
    of individual connected components while normalizing by the total node count 
    of the original graph.

    Args:
        G_nx (nx.Graph): NetworkX graph object.

    Returns:
        float: Global efficiency value (0.0 to 1.0).
    """
    if G_nx.number_of_nodes() < 2:
        return 0.0

    # Convert to igraph for performance
    G_ig, _, _ = convert_nx_to_ig(G_nx)

    total_inverse_path_length = 0.0

    # Iterate over connected components ("Islands")
    # This optimization avoids infinite distances in disconnected graphs
    for component in G_ig.components():
        if len(component) < 2:
            continue

        subgraph_ig = G_ig.subgraph(component)

        # Calculate shortest paths within the component
        path_lengths_matrix = subgraph_ig.shortest_paths(weights='weight')

        # Sum of 1/d_ij for all pairs where d_ij > 0
        total_inverse_path_length += sum(
            1.0 / d for row in path_lengths_matrix for d in row if d > 0 and d != float('inf')
        )

    # The denominator is based on the TOTAL number of nodes in the original graph
    n_total = G_nx.number_of_nodes()
    denominator = n_total * (n_total - 1)

    return total_inverse_path_length / denominator if denominator > 0 else 0.0


class EconomicCalculator:
    """
    Handles the calculation of Direct Economic Loss (Physical Damage) and 
    Indirect Economic Loss (Input-Output Model inefficiency).
    """

    def __init__(self, io_df, unit_cost):
        """
        Args:
            io_df (pd.DataFrame): Input-Output table for economic sectors.
            unit_cost (float): Cost per unit length for physical repairs.
        """
        self.unit_cost = unit_cost
        self.io_data = self._prepare_io(io_df)

        # Identify the Transport sector index (assuming column name contains 'Transport')
        # Adjust '交通运输' to English 'Transportation' if your excel headers change, 
        # or keep the Chinese key if the source data remains in Chinese.
        try:
            self.sector_idx = self.io_data['sectors'].index("交通运输")
        except ValueError:
            # Fallback or specific error if the specific column name isn't found
            # Assuming the user might change data headers, we warn or fail.
            logging.warning("Sector '交通运输' not found. Please ensure IO table has the correct index.")
            self.sector_idx = 0

            # Baseline cache
        self.eff_benchmark = None
        self.eff_original = None

        logging.info("Economic Calculator initialized.")

    def set_baseline_efficiencies(self, eff_benchmark, eff_original):
        """
        Stores pre-computed baseline efficiencies to avoid redundant calculations.

        Args:
            eff_benchmark (float): Efficiency of the optimal (no-flood) network.
            eff_original (float): Efficiency of the flooded network (no action).
        """
        self.eff_benchmark = eff_benchmark
        self.eff_original = eff_original
        logging.info(
            f"Baselines set :: Benchmark Eff={self.eff_benchmark:.6f}, Original Flood Eff={self.eff_original:.6f}")

    def _prepare_io(self, io_df):
        """
        Prepares the Leontief Inverse Matrix from the Input-Output table.
        """
        try:
            # Filter out summary columns to get the inter-sector flow matrix
            sectors = [col for col in io_df.columns if col not in ['中间使用合计', '最终使用合计', '总产出']]

            # Calculate Technical Coefficient Matrix (A)
            # A_ij = Flow_ij / Total_Output_j
            total_output = io_df.loc[sectors, '总产出'].values
            A = io_df.loc[sectors, sectors].values / total_output.clip(min=1e-6)
            A = np.nan_to_num(A, nan=0)

            # Calculate Leontief Inverse: L = (I - A)^-1
            leontief_inverse = np.linalg.inv(np.eye(len(sectors)) - A)

            return {
                'sectors': sectors,
                'leontief_inverse': leontief_inverse,
                'final_demand': io_df.loc[sectors, '最终使用合计'].values
            }
        except Exception as e:
            logging.error(f"IO Table preparation failed: {e}")
            raise

    def direct_loss(self, G):
        """
        Calculates direct physical damage cost based on flood depth and road length.
        """
        total = 0.0
        for _, _, data in G.edges(data=True):
            if 'Floodinfo' not in data or data.get('construction_cost', 0) == 0:
                continue

            # Iterate through segmented flood depths on the edge
            for depth, length in data['Floodinfo'].items():
                total += self.unit_cost * length * self._loss_probability(depth)
        return total

    def indirect_loss(self, G_updated):
        """
        Wrapper to return only the total indirect loss.
        """
        _, total_loss = self.get_indirect_loss_details(G_updated)
        return total_loss

    def get_indirect_loss_details(self, G_updated):
        """
        Calculates indirect economic loss using the Inoperability Input-Output Model (IIM).
        Loss is driven by the drop in network efficiency relative to the benchmark.

        Returns:
            tuple: (dict of sector-wise losses, float total indirect loss)
        """
        if self.eff_benchmark is None:
            raise ValueError("Baseline efficiency not set. Call set_baseline_efficiencies() first.")

        eff_updated = _network_efficiency_ig_exact(G_updated)

        # Calculate the 'shock' (perturbation) to the transport sector
        shock = self.eff_benchmark - eff_updated
        if shock < 0:
            shock = 0

        # Create shock vector (only Transport sector is directly affected by network efficiency)
        shock_vector = np.zeros(len(self.io_data['sectors']))
        shock_vector[self.sector_idx] = shock * self.io_data['final_demand'][self.sector_idx]

        # Calculate output change across all sectors via Leontief Inverse
        # Delta_X = L * Delta_f
        output_change = self.io_data['leontief_inverse'] @ shock_vector
        total_indirect_loss = np.sum(output_change)

        # Map results to sector names
        loss_details = dict(zip(self.io_data['sectors'], output_change))

        return loss_details, total_indirect_loss

    def _loss_probability(self, depth):
        """
        Returns the damage ratio (0.0 to 1.0) based on flood depth (meters).
        Based on standard depth-damage curves for road infrastructure.
        """
        if depth <= 0:
            return 0.0
        elif depth < 0.5:
            return 0.72 * depth
        elif depth < 1:
            return 0.42 * depth + 0.15
        elif depth < 1.5:
            return 0.32 * depth + 0.25
        elif depth < 2:
            return 0.24 * depth + 0.37
        elif depth < 3:
            return 0.15 * depth + 0.55
        else:
            return 1.0