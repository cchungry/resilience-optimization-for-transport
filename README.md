# Multi-Objective Road Network Resilience Optimization Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## Overview

This repository contains a comprehensive framework for optimizing road network resilience against flood disasters. The model employs a **Multi-Objective Optimization** approach to balance economic efficiency and social equity (accessibility) under limited budget constraints.

The core engine utilizes a hybrid **Heuristic Genetic Algorithm (GA)** integrated with **Variable Neighborhood Descent (VND)** and **Simulated Annealing (SA)** to identify optimal retrofitting strategies (Maintenance vs. Upgrade) for critical road segments.

## Key Features

* **Multi-Dimensional Evaluation:**
    * **Economic Loss:** Calculates Direct Physical Damage and Indirect Economic Loss using an **Inoperability Input-Output Model (IIM)**.
    * **Social Equity:** Evaluates Accessibility to critical services weighted by population/social importance (ZHS).
* **High-Performance Computing:**
    * Leverages `igraph` for ultra-fast shortest path and network efficiency calculations.
    * Supports parallel processing for large-scale network evaluations.
* **Hybrid Optimization Algorithms:**
    * **Heuristic GA:** Initialization and mutation operators guided by **Betweenness Centrality**.
    * **Local Search:** Integrated VND and SA modules to refine elite solutions.
    * **Memetic Strategies:** Implements "Destruction & Reconstruction" immigration operators to prevent premature convergence.
* **Flexible Configuration:** Fully configurable via YAML files to switch between algorithms (`standard_ga`, `heuristic_ga_vnd`, `sa_only`, etc.) and adjust objective weights.

## Project Structure

```text
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
