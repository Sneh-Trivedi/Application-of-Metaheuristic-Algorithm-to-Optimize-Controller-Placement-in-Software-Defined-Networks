#Standard Genetic Algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # For calculating Euclidean distance
import random

# Settings for reproducibility
# np.random.seed(0)
# random.seed(0)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r


nodes = {
    "0": {"Latitude": 51.46117, "Longitude": -0.93083, "label": "TVN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "1": {"Latitude": 51.37795, "Longitude": -2.35907, "label": "SWERN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "2": {"Latitude": 52.5, "Longitude": -1.96667, "label": "MidMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "3": {"Latitude": 51.48, "Longitude": -3.18, "label": "WREN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "4": {"Latitude": 51.50853, "Longitude": -0.12574, "label": "LMN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "5": {"Latitude": 52.2, "Longitude": 0.11667, "label": "EastNet","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "6": {"Latitude": 50.90395, "Longitude": -1.40428, "label": "LeNSE","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "7": {"Latitude": 51.38914, "Longitude": 0.54863, "label": "Kentish MAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "8": {"Latitude": 52.83111, "Longitude": -1.32806, "label": "EMMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "10": {"Latitude": 55.95, "Longitude": -3.2, "label": "Sco-locate","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "11": {"Latitude": 51.45625, "Longitude": -0.97113, "label": "Reading","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "12": {"Latitude": 53.79648, "Longitude": -1.54785, "label": "Leeds","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "13": {"Latitude": 53.39254, "Longitude": -2.58024, "label": "Warrington","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "14": {"Latitude": 55.86515, "Longitude": -4.25763, "label": "Glasgow","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "15": {"Latitude": 51.50853, "Longitude": -0.12574, "label": "Telehouse","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "16": {"Latitude": 51.50853, "Longitude": -0.12574, "label": "Telecity","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "17": {"Latitude": 51.50853, "Longitude": -0.12574, "label": "London","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "18": {"Latitude": 51.45, "Longitude": -2.58333, "label": "Bristol","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "19": {"Latitude": 54.58333, "Longitude": -2.83333, "label": "C&NLMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "20": {"Latitude": 54.5, "Longitude": -6.5, "label": "NIRAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "21": {"Latitude": 57.12, "Longitude": -4.71, "label": "UHIMI","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "22": {"Latitude": 57.14369, "Longitude": -2.09814, "label": "AbMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "23": {"Latitude": 56.33871, "Longitude": -2.79902, "label": "FaTMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "24": {"Latitude": 55.95, "Longitude": -3.2, "label": "EaStMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "25": {"Latitude": 55.86515, "Longitude": -4.25763, "label": "Clydenet","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "26": {"Latitude": 54.97328, "Longitude": -1.61396, "label": "NorMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "27": {"Latitude": 54.27966, "Longitude": -0.40443, "label": "YHMAN","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)},
    "28": {"Latitude": 53.48095, "Longitude": -2.23743, "label": "NNW","cost": np.random.randint(100, 500), "capacity": np.random.randint(10, 20)}
}
# Parameters
n_switches = 6  # Number of switches
n_potential_sites = len(nodes)  # Number of potential sites for controllers
population_size = 10  # Number of solutions in the population
generations = 5  # Number of generations
mutation_rate = 0.1  # Probability of mutation



# Extracting coordinates, costs, and capacities
site_coords = np.array([(node["Latitude"], node["Longitude"]) for node in nodes.values()])
site_costs = np.array([node["cost"] for node in nodes.values()])
site_capacities = np.array([node["capacity"] for node in nodes.values()])

switches = np.array([(np.random.uniform(40, 70), np.random.uniform(-7, 4)) for _ in range(n_switches)])
switch_loads = np.array([np.random.randint(10, 20) for _ in range(n_switches)])

# Print initial switch loads, site costs, and capacities
print("Switch loads:", switch_loads)
print("Site costs:", site_costs)
print("Site capacities:", site_capacities)

# Distance matrix between each switch and potential controller site
dist_matrix = np.zeros((n_switches, n_potential_sites))
for i, switch in enumerate(switches):
    for j, site in enumerate(site_coords):
        dist_matrix[i, j] = haversine(switch[1], switch[0], site[1], site[0])

def select_two_members(population):
    return random.sample(population, 2)

def custom_generation(m1, m2, n_potential_sites, n_switches):
    # This will be your custom generation function based on your description
    # For now, I'm leaving it similar to the crossover
    crossover_point = random.randint(1, n_switches - 1)
    print(f"Crossover point: {crossover_point}")
    return np.concatenate((m1[:crossover_point], m2[crossover_point:]))

def replace_worst_member(population, fitness, new_member, new_fitness):
    worst_fitness_index = np.argmax(fitness)
    if new_fitness < fitness[worst_fitness_index] and new_fitness not in fitness:
        print(f"Replacing individual {worst_fitness_index} with new member due to better fitness.")

        population[worst_fitness_index] = new_member
        fitness[worst_fitness_index] = new_fitness
    return population, fitness

def evaluate_solution(solution, dist_matrix, switch_loads, site_costs, site_capacities):
    """
    Evaluate the cost of a solution based on the location of controllers and connections between nodes and controllers.
    """
    total_cost = 0
    total_capacity_used = np.zeros(n_potential_sites)
    
    for switch_idx, switch_load in enumerate(switch_loads):
        closest_site_idx = solution[switch_idx]
        if total_capacity_used[closest_site_idx] + switch_load <= site_capacities[closest_site_idx]:
            total_cost += dist_matrix[switch_idx, closest_site_idx] * switch_load
            total_capacity_used[closest_site_idx] += switch_load
        else:
            total_cost += 1e6  # Large penalty for exceeding capacity
        
    # Add fixed costs of used sites
    for site_idx, site_cost in enumerate(site_costs):
        if total_capacity_used[site_idx] > 0:
            total_cost += site_cost

    # Debug: Print total capacity used vs capacity limits
    for i, (used, capacity) in enumerate(zip(total_capacity_used, site_capacities)):
        if used > capacity:
            print(f"CAPACITY EXCEEDED AT SITE {i}: used {used}, capacity {capacity}")
# This would show you if at any point you are assigning more load to a site than it can handle.

    
    return total_cost

def initialize_population(n_switches, n_potential_sites, population_size):
    """
    Initialize a population of solutions, each solution is a mapping of switches to controller sites.
    """
    population = [np.random.randint(0, n_potential_sites, n_switches) for _ in range(population_size)]
    print("\nInitial population:")
    for i, individual in enumerate(population):
        print(f"Individual {i + 1}: {individual}")
    return population

# def genetic_algorithm(n_switches, n_potential_sites, generations, population_size, mutation_rate):
#     # Initialize population
#     population = initialize_population(n_switches, n_potential_sites, population_size)
#     best_cost = np.inf
#     best_solution = None
#     cost_history = []  # Keep track of the best cost in each generation
    
#     for generation in range(generations):
#         # Evaluate population
#         fitness = [evaluate_solution(solution, dist_matrix, switch_loads, site_costs, site_capacities) 
#                    for solution in population]
        
#         # Selection
#         sorted_indices = np.argsort(fitness)
#         population = [population[idx] for idx in sorted_indices[:population_size // 2]]  # Select top half
        

#         # Print out the fitness of the current generation
#         print(f'\nGeneration {generation}: Best fitness = {fitness[sorted_indices[0]]}, Worst fitness = {fitness[sorted_indices[-1]]}')
#         print("Selected individuals:")
#         for i, individual in enumerate(population):
#             print(f"Individual {i + 1}: {individual}, Fitness: {fitness[sorted_indices[i]]}")
        
#         # Crossover (Generate new solutions by combining parts of best solutions)
#         new_generation = []
#         while len(new_generation) < population_size - len(population):
#             parents = random.sample(population, 2)
#             crossover_point = random.randint(1, n_switches-1)
#             child = np.concatenate([parents[0][:crossover_point], parents[1][crossover_point:]])
#             new_generation.append(child)

#         print("New generation after crossover and before mutation:")
#         for i, individual in enumerate(new_generation):
#             print(f"Individual {i + 1}: {individual}")
        
#         # Mutation
#         for individual in new_generation:
#             if random.random() < mutation_rate:
#                 mutate_point = random.randint(0, n_switches-1)
#                 individual[mutate_point] = random.randint(0, n_potential_sites-1)
#                 print(f"Individual after mutation: {individual}")

#         population.extend(new_generation)
        
#         # Update best solution
#         current_best_cost = fitness[sorted_indices[0]]
#         if current_best_cost < best_cost:
#             best_cost = current_best_cost
#             best_solution = population[0]
#             print(f'Generation {generation}: New Best Cost = {best_cost}')
#         else:
#             print(f'Generation {generation}: No Improvement')
        
#         cost_history.append(best_cost)
    
#     return best_solution, best_cost, cost_history

def genetic_algorithm(n_switches, n_potential_sites, generations, population_size, mutation_rate):
    # Initialize population
    population = initialize_population(n_switches, n_potential_sites, population_size)
    fitness = [evaluate_solution(individual, dist_matrix, switch_loads, site_costs, site_capacities) for individual in population]
    best_cost = np.inf
    best_solution = None
    cost_history = []  # Keep track of the best cost in each generation
    
    for generation in range(generations):
        # Print out the fitness of the current generation
        print(f"\nGeneration {generation}")
        print(f'\nGeneration {generation}: Best fitness = {min(fitness)}, Worst fitness = {max(fitness)}')
        
        # Selection
        m1, m2 = select_two_members(population)
        print(f"Selected for breeding: {m1}, {m2}")
        # Custom generation function
        new_member = custom_generation(m1, m2, n_potential_sites, n_switches)
        new_member_fitness = evaluate_solution(new_member, dist_matrix, switch_loads, site_costs, site_capacities)

        print(f"Generated new member through crossover: {new_member}, Cost: {new_member_fitness}")

        # Mutation
        if random.random() < mutation_rate:
            mutate_point = random.randint(0, n_switches-1)
            new_member[mutate_point] = random.randint(0, n_potential_sites-1)
        
        # Evaluate new member
        new_fitness = evaluate_solution(new_member, dist_matrix, switch_loads, site_costs, site_capacities)
        
        # Replacement
        population, fitness = replace_worst_member(population, fitness, new_member, new_fitness)
        
        # Update best solution if the new solution is better than the current best
        best_idx = np.argmin(fitness)  # Find the index of the best fitness
        if fitness[best_idx] < best_cost:
            best_cost = fitness[best_idx]
            best_solution = population[best_idx]
            print(f'Generation {generation}: New Best Cost = {best_cost}')
        else:
            print(f'Generation {generation}: No Improvement')
        
        cost_history.append(best_cost)
    
    return best_solution, best_cost, cost_history


# Run the genetic algorithm
best_solution, best_cost, cost_history = genetic_algorithm(n_switches, n_potential_sites, generations, population_size, mutation_rate)
unique_controllers = set(best_solution)
total_fixed_cost = sum(site_costs[i] for i in unique_controllers)
total_routing_cost = 0
# Count how many switches are connected to each controller and calculate the routing cost
controller_counts = {site: 0 for site in unique_controllers}
for switch_index, site_index in enumerate(best_solution):
    if site_index in unique_controllers:  # Check if this switch is connected to a unique controller
        total_routing_cost += dist_matrix[switch_index, site_index] * switch_loads[switch_index]
        controller_counts[site_index] += 1
# Output results
print("\nFinal results:")
print(f"Total number of controllers used: {len(unique_controllers)}")
print(f"Indices of controllers placed: {sorted(unique_controllers)}")
print(f"Fixed costs of placed controllers: {[site_costs[i] for i in unique_controllers]}")
print(f"Total fixed cost: {total_fixed_cost}")
print(f"Total routing cost: {total_routing_cost:.2f}")
print(f"Final Best Solution: {best_solution}, Final Best Cost: {best_cost}")


# Visualization of final solution
plt.figure(figsize=(12, 8))
# Plot all potential sites (circles)
for i, (lat, lon) in enumerate(site_coords):
    plt.scatter(lon, lat, facecolors='none', edgecolors='b', s=100, label='Potential Site' if i == 0 else "")
    plt.text(lon, lat - 3, f'C{i+1}', fontsize=9, ha='center')

# Plot all switches ('x')
for i, (lat, lon) in enumerate(switches):
    plt.scatter(lon, lat, color='black', marker='x', label='Switch' if i == 0 else "")
    plt.text(lon, lat + 3, f'S{i+1}', fontsize=9, ha='center')

# Draw connections and mark active controllers with a red triangle
for switch_index, site_index in enumerate(best_solution):
    switch = switches[switch_index]
    site = site_coords[site_index]
    plt.plot([switch[1], site[1]], [switch[0], site[0]], 'gray', linestyle='--')
    plt.scatter(site[1], site[0], color='red', marker='^', s=50)  # Red triangle for active controller

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Network Design with Genetic Algorithm')
plt.legend()
plt.show()
