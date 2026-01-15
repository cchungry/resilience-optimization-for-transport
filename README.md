Multi-Objective Road Network Resilience Optimization Framework
Overview
This repository contains a comprehensive framework for optimizing road network resilience against flood disasters. The model employs a Multi-Objective Optimization approach to balance economic efficiency and social equity (accessibility) under limited budget constraints.

The core engine utilizes a hybrid Heuristic Genetic Algorithm (GA) integrated with Variable Neighborhood Descent (VND) and Simulated Annealing (SA) to identify optimal retrofitting strategies (Maintenance vs. Upgrade) for critical road segments.

Key Features
Multi-Dimensional Evaluation:

Economic Loss: Calculates Direct Physical Damage and Indirect Economic Loss using an Inoperability Input-Output Model (IIM).

Social Equity: Evaluates Accessibility to critical services weighted by population/social importance (ZHS).

High-Performance Computing:

Leverages igraph for ultra-fast shortest path and network efficiency calculations.

Supports parallel processing for large-scale network evaluations.

Hybrid Optimization Algorithms:

Heuristic GA: Initialization and mutation operators guided by Betweenness Centrality.

Local Search: Integrated VND and SA modules to refine elite solutions.

Memetic Strategies: Implements "Destruction & Reconstruction" immigration operators to prevent premature convergence.

Flexible Configuration: Fully configurable via YAML files to switch between algorithms (standard_ga, heuristic_ga_vnd, sa_only, etc.) and adjust objective weights.

Project Structure
Plaintext

├── config/
│   └── config.yaml          # Central configuration file (Algorithm mode, Budget, Weights)
├── data/                    # Input datasets (Roads, IO Table, Social Data)
├── output/                  # Results (Pareto front, Logs, Debug info)
├── accessibility.py         # Accessibility calculation logic (igraph optimized)
├── economic.py              # Economic loss models (Direct + Input-Output)
├── graph_utils.py           # NetworkX to igraph conversion utilities
├── main.py                  # Entry point of the application
├── network.py               # Network topology construction and strategy application
├── optimization.py          # Core GA/SA/VND algorithm implementations
└── vnd.py                   # Variable Neighborhood Descent local search logic
Installation
Prerequisites
Python 3.8 or higher

C compiler (required for python-igraph in some environments)

Dependencies
Install the required Python packages using pip:

Bash

pip install numpy pandas networkx python-igraph pymoo pyyaml tqdm openpyxl
(Note: Depending on your OS, installing python-igraph might require system-level dependencies. Refer to the igraph documentation if you encounter issues.)

Usage
Prepare Data: Ensure your input files (roads.xlsx, io_table.xlsx, 000_livingarea_ZHS.xlsx) are placed in the data/ directory.

Configure: Edit config/config.yaml to set your desired parameters (Budget, Algorithm Mode, Parallel Workers).

Run: Execute the main script:

Bash

python main.py
Results: Results will be saved in the output/ directory, including:

000_pareto_front.csv: The best solutions found.
000_gen_history_*.csv: Optimization progress logs.
000_io_loss.csv & 000_A_nodes_o.csv: Detailed breakdown of the best solution.

Configuration (config.yaml)
The framework behavior is controlled by config/config.yaml. Key settings include:

algorithm_mode:

heuristic_ga_vnd: (Recommended) GA with Centrality Heuristics + VND Local Search.

standard_ga: Vanilla Genetic Algorithm.

sa_only: Standalone Simulated Annealing.

budget: Total financial constraint for road maintenance/upgrading.

weights: Balance between Economic (a) and Social (b) objectives.

YAML

# Example Config Snippet
algorithm_mode: 'heuristic_ga_vnd'
budget: 5000
parallel:
  max_workers: 8
vnd_settings:
  enable: true
  run_every: 5
Data Requirements
The model expects the following input files in the data/ folder:

roads.xlsx: Road network edges with columns for ID, IN_ID, OUT_ID, Length, original_time, flood depth (Depth_ssp2_20a), and flood info JSON (Floodinfo_...).

io_table.xlsx: Economic Input-Output table for indirect loss calculation.

000_livingarea_ZHS.xlsx: Node weights representing social importance (e.g., population or service capacity).

Citation
If you use this code for your research, please cite:

[Your Name / Paper Title Placeholder] (If applicable, insert DOI or link to your publication here)

License
This project is licensed under the MIT License - see the LICENSE file for details.