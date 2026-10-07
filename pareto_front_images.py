import pickle
import os
import matplotlib.pyplot as plt

# Directory where Pareto front images will be saved
output_dir = '<test_dir>/pareto_fronts'
os.makedirs(output_dir, exist_ok=True)

# Helper function to extract fitness values from a given dataset
def extract_fitness_values(dataset):
    fitness_values = []
    for attributes in dataset.values():
        if 'fitness' in attributes:
            if attributes['fitness'] is not None and attributes['fitness'] != (-float('inf'), float('inf')):
                fitness_values.append(attributes['fitness'])
    return fitness_values

# Function to load data and extract fitness values for a given generation
def get_fitness_values_for_generation(gen_number):
    file_path = f'<test_dir>/global_data/global_gen_{gen_number}.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract fitness values from GLOBAL_DATA and GLOBAL_DATA_HIST
    global_data_fitness = extract_fitness_values(data.get('GLOBAL_DATA'))
    global_data_hist_fitness = extract_fitness_values(data.get('GLOBAL_DATA_HIST'))

    return global_data_fitness, global_data_hist_fitness

# Function to identify Pareto frontier
def pareto_frontier(rates):
    # Sort by first objective in descending order and second objective in ascending order
    sorted_rates = sorted(rates, key=lambda x: (-x[0], x[1]))
    
    pareto_front = [sorted_rates[0]]
    for rate in sorted_rates[1:]:
        if rate[1] < pareto_front[-1][1]:
            pareto_front.append(rate)
    return pareto_front

# Function to create and save Pareto front plot for a generation
def plot_pareto_front(global_data_fitness, global_data_hist_fitness, gen_number):
    output_path = os.path.join(output_dir, f'pareto_gen_{gen_number}.png')
    if os.path.exists(output_path):
        print(f"Pareto front image for generation {gen_number} already exists.")
        return
    # Combine fitness values from both datasets
    all_fitness_values = global_data_fitness + global_data_hist_fitness
    
    if not all_fitness_values:
        print(f"No valid fitness values for generation {gen_number}")
        return
    
    # Extract x and y coordinates for global data fitness
    global_xs, global_ys = zip(*global_data_fitness) if global_data_fitness else ([], [])
    
    # Extract x and y coordinates for global data hist fitness
    hist_xs, hist_ys = zip(*global_data_hist_fitness) if global_data_hist_fitness else ([], [])
    
    # Determine the Pareto front
    pareto_front = pareto_frontier(all_fitness_values)
    pareto_xs, pareto_ys = zip(*pareto_front) if pareto_front else ([], [])
    print(f"Pareto front for generation {gen_number}: {pareto_front}")
    # Get points for the Pareto frontier line
    pareto_xs = list(pareto_xs)
    pareto_ys = list(pareto_ys)

    # Create the plot with a stretched x-axis through the figsize parameter
    plt.figure(figsize=(10, 5))  # Width is 10 and height is 5
    plt.scatter(global_xs, global_ys, s=10, alpha=0.5, label='Global Data Points', color='blue')
    plt.scatter(hist_xs, hist_ys, s=10, alpha=0.5, label='Hist Data Points', color='green')
    plt.scatter(pareto_xs, pareto_ys, s=15, color='red', label='Pareto Front Points')

    # Add a star marker at the specified coordinates
    special_x = 0.9252
    special_y = 518230.0
    plt.scatter([special_x], [special_y], color='gold', s=100, marker='*', label='ExquisiteNetV2')

    if pareto_xs and pareto_ys:
        pareto_xs = [1] + pareto_xs + [0]
        pareto_ys = [1e7] + pareto_ys + [0]

    plt.step(pareto_xs, pareto_ys, color='gray', where='post', label='Pareto Front Line')
    plt.xlabel('Accuracy')
    plt.ylabel('Parameters')
    plt.title(f'Pareto Front - Generation {gen_number}')
    plt.legend()
    
    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    
    # Set x and y axis limits
    plt.xlim(0.8, 1)
    plt.ylim(1e4, 1e7)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'pareto_gen_{gen_number}.png')
    plt.savefig(output_path)
    plt.close()

# Create and save Pareto front images for each generation from 1 to 27
for gen_number in range(1, 1000):
    global_data_fitness, global_data_hist_fitness = get_fitness_values_for_generation(gen_number)
    if global_data_fitness or global_data_hist_fitness:  # Only plot if there are fitness values to plot
        plot_pareto_front(global_data_fitness, global_data_hist_fitness, gen_number)
    print(f'Pareto front for generation {gen_number} processed.')

print('Pareto front images for all generations have been created and saved.')