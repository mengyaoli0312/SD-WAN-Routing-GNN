# GNN-Enhanced Routing in SD-WAN

This repository contains the implementation and experimental code for our work on
GNN-enhanced intelligent routing in an SD-WAN environment integrating terrestrial
and LEO satellite networks.

The framework constructs a hybrid terrestrial-satellite network from satellite TLE
data, ground locations, and measured link/network conditions, and applies a
Deep Q-Network (DQN) with a Graph Neural Network (GNN)-based state representation
to select routing paths under heterogeneous latency, bandwidth, and link conditions.

## Overview

The main workflow of the repository includes:

1. Satellite orbit propagation from TLE data using SGP4.
2. Construction of a hybrid network topology containing:
   - Inter-Satellite Links (ISLs),
   - Satellite-to-Ground links,
   - Terrestrial fiber links.
3. Integration of measured/sampled network and weather-related link characteristics.
4. Generation of candidate terrestrial, satellite, and hybrid routing paths.
5. GNN-based representation of network-link states.
6. DQN-based path selection and routing optimization.
7. Comparison with conventional and heuristic routing baselines.

The current implementation uses 66 LEO satellites and multiple ground locations,
including Enschede, Osnabrück, and several representative cities worldwide.

## Repository Structure

```text
.
├── tle_loader.py
├── topology_builder.py
├── train_DQN.py
├── environment1.py
├── mpnn.py
│
├── baseline2.py
├── baseline3.py
│
├── analysis_data_Enschede.csv
├── analysis_data_Osnabrück.csv
│
├── iridium_66_main.tle
│
├── fixed_reqs1.jsonl
├── fixed_reqs2.jsonl
├── fixed_reqs3.jsonl
│
├── baseline_ground_results.jsonl
├── baseline1222.jsonl
├── baseline1291.jsonl
│
├── daily_24h_all_rain_check.csv
├── daily_24h_clear_check.csv
│
└── Logs/
