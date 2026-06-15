# CogniCity: A Multi-Agent Framework for Assessing the Systemic Impact of Electric Mobility in Urban Environments
This repository contains the core implementation of an interdisciplinary, agent-based simulation framework designed to evaluate the coupled impacts of electric mobility on both urban geographical spaces and electrical distribution grids.

## Overview
Unlike typical heavy data-driven simulation platforms, this framework is intentionally designed to operate under data-constrained conditions (data-scarce environments). It provides urban planners and local authorities with a scalable and accessible decision-support tool. Instead of requiring massive "Big Data" infrastructure, the model utilizes easily sourceable geographic network data paired with aggregate local demographic variables and small-scale routine surveys to simulate realistic daily mobility habits and their direct technical impacts on energy infrastructure.

## Key Features
- **Low Data Overhead:** Highly optimized for smaller municipalities or data-restricted regions.
- **Interdisciplinary Coupling:** Seamlessly bridges individual behavioral mobility choices with technical grid resilience assessment.
- **Overlapping Environments:** Simulates coexisting layers—a geographical system (where agents move) and an electrical power network (where energy demand is injected).

## Model Architecture (Ontological Layer)
The system is divided into active decision-making entities and structural passive elements:

### Active Entities
- **Citizens:** The core decision-making agents. They are categorized into five distinct demographic archetypes (Adult Male, Adult Female, Child, Elderly, Youth) which structurally dictate their level of autonomy, walking speeds, and general daily priorities.
- **Vehicles:** Dynamic instrumental objects used by citizens. Their mobility tracks geographic paths while their charging states project instantaneous power loads directly into the electrical environment. Supported modes include private internal combustion engine cars (petrol/diesel/CNG), electric cars (EVs), urban buses (diesel/CNG/electric), and pedestrian movement.

### Passive Entities
- **Points of Interest (POIs):** Extracted from OpenStreetMap (OSM) tags and mapped according to the "15-minute city" framework. They are classified into 6 functional archetypes (Residential, Work, Study, Entertainment, Duties, Healthcare) plus transit infrastructure (parking spaces, public transport stops, and EV charging stations).
- **Electrical Nodes:** Fixed entities within the distribution network that absorb charging power requirements, allowing real-time tracking of voltage fluctuations and peak congestion.

![Visual description of the conceptual functioning of the model.](https://drive.google.com/uc?export=view&id=1nD_RJ4hpzsAtVqobNiAfdYQ3o0yg789o)

## Procedural Execution Flow
The simulation engine executes sequentially across 7 structural phases:

- **Phase 0:** System Integrity Verification: Automatically validates the formatting and availability of critical profile datasets, behavioral files, and configuration files.
- **Phase 1:** Environment Initialization: Dynamically generates the spatial geometry (roads and POI coordinates from OSM data) alongside the logically linked electrical node mesh.
- **Phase 2:** Synthetic Population Generation: Hierarchically spawns individual citizens and bundles them into compatible households (e.g., nuclear family, single parenthood, unipersonal home, DINKs) using aggregate joint probabilities. Vehicles are then distributed at the household level.
- **Phase 3:** Daily TODO List Generation: Assigns daily activity sequences to each agent governed by their respective life-stage behavioral archetypes.
- **Phase 4:** Planning & Transport Choice Modelling: Households evaluate route combinations. Mode choice is realized via a distance-based fuzzy probabilistic framework that dynamically arbitrates between active mobility, public transport, or private vehicle use.
- **Phase 5:** Schedule & Grid Demand Generation: Steps through the time-series simulation where agents execute their mobility diaries. EV charging demands (modeled via localized charging paradigms like midday solar vs. overnight baseload charging) are computed and mapped directly to coupled electrical nodes.
- **Phase 6:** Aggregation of Results: Compiles, logs, and exports systemic analytical outputs, including CO2 emissions, travel time profiles, and electrical load curves.

![Visual representation of the system’s flow.](https://drive.google.com/uc?export=view&id=1CaTXGl7d8xyZ8N9pE2s-tV2cC85_GozV)

## Technical Specifications
- **Core Language:** Python (version 3.12).
- **Geospatial Processing:** Automated parsing of network architectures from OpenStreetMap (OSM).

## Publication & Further Information
This README.md covers the fundamental architecture, file dependencies, and execution routines of the repository. For all deep technical specificities, please refer to our official paper.

Please consult the paper for details regarding:
- The full mathematical formalization of the distance-dependent fuzzy logic modal choice models.
- Stochastic calibration tables for household demographic pairing and socio-economic profiling (ESeC/UNSD-aligned).
- The underlying power flow assumptions and EV battery charging curve equations.
- Full empirical results and stochastic stability validations from our primary case study: Kanaleneiland, Utrecht (The Netherlands).

## Acknowledgements
This project has been funded by Grant PCI2023-145951-2, which is supported by MCIN/AEI/ 10.13039/501100011033 under the Driving Urban Transitions Partnership (F-DUT-2022-0241), which is co-funded by the European Commission. The authors acknowledge the technical and human support that the DIPC Supercomputing Centre provided.

![funders.](https://drive.google.com/uc?export=view&id=10qPfLBYbNe9IBZWhyY7kD_2puwYAQOxo)
