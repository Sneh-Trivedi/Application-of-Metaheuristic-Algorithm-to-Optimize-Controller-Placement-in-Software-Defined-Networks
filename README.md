# Application-of-Metaheuristic-Algorithm-to-Optimize-Controller-Placement-in-Software-Defined-Networks

This repository contains the implementation of a Genetic Algorithm (GA) and a Cost-Capacity Constrained Greedy Placement Algorithm (CCCGP) for solving the Controller Placement Problem (CPP) in Software-Defined Networks (SDNs). The aim of this project is to optimize the placement of controllers in a network to minimize costs and improve network efficiency.

## Abstract

The CPP is an NP-hard problem that involves determining the strategic placement of controllers within an SDN to reduce latency and routing costs while ensuring scalability and performance. This project introduces a novel Genetic Algorithm designed to outperform traditional methods, offering significant cost savings and improved controller placements.

The algorithms have been tested on real-world network topologies, such as the Janet Backbone, using distance metrics calculated via the Haversine formula. Results show that the proposed GA achieves approximately 33.33% lower costs compared to the CCCGP for larger networks, demonstrating its superiority in managing complex topologies.

## Features

Random Greedy Algorithm (CCCGP): A baseline approach that prioritizes cost and capacity constraints.
Genetic Algorithm (GA): Employs selection, crossover, and mutation operators to iteratively evolve optimal controller placements.
Real-World Topology Support: Experiments conducted on the Janet Backbone network topology.
Parameter Optimization: Supports customization of parameters like population size, mutation rate, and number of controllers.
Efficient Cost Minimization: Reduces fixed and operational costs while optimizing propagation latency.
Algorithms

## 1. Cost-Capacity Constrained Greedy Placement Algorithm (CCCGP)
A greedy approach that iteratively selects controller placement sites based on cost and capacity constraints. While efficient for smaller configurations, it often converges to local optima.

## 2. Genetic Algorithm (GA)
Inspired by natural selection, the GA evolves a population of potential solutions through:

Selection: Randomly selects parent solutions for crossover.
Crossover: Combines traits from parent solutions to create offspring.
Mutation: Introduces randomness to avoid local optima.
Key Parameters:
Population Size: 20
Mutation Rate: 0.1
Generations: 500
Network Topology: Janet Backbone (29 nodes)

## Performance
Cost Reduction: Achieves up to 33.33% cost savings compared to CCCGP.
Optimal Controller Placements: Identifies optimal configurations for networks with 50 and 100 switches.
Results

Objective Function Performance: Shows a balance between fixed costs and propagation latency as the number of controllers increases.
Execution Time Analysis: The GA exhibits near-linear growth in execution time with the number of controllers.
Optimal Locations: Visualizations display controller placements that minimize costs and latency.

### This implementation is based on the research paper titled "Application of Metaheuristic Algorithms to Optimize Controller Placement in Software-Defined Networks (SDNs)" by Sneh Trivedi et al. For detailed insights, refer to the publication on [ResearchGate](https://www.researchgate.net/publication/383824472_Application_of_Metaheuristic_Algorithms_to_Optimize_Controller_Placement_in_Software-Defined_Networks_SDNs).
