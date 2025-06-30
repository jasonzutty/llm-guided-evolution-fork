import pickle
import os
import sys
import os

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from run_improved import load_checkpoint

import numpy as np
from pareto import eps_sort
import glob
import matplotlib.pyplot as plt
import csv
import pickle
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Original, restore later
# map_obj={0: {'name':'parameter', 'weight': 'False' },
#          1: {'name':'inference_speed', 'weight': 'False' },
#          2: {'name':'precision', 'weight': 'True' },
#          3: {'name':'recall', 'weight': 'True' },
#          4: {'name':'mAP50', 'weight': 'True' },
#          5: {'name':'mAP50-90', 'weight': 'True' },
#         #  6: {'name':'gflops', 'weight': 'True' },
#         #  7: {'name':'fps', 'weight': 'True' },
#         #  8: {'name':'latency', 'weight': 'False' },
#         #  9: {'name':'AssA', 'weight': 'True' },
# }


map_obj={0: {'name':'mAP50-90', 'weight': 'True' },
         1: {'name':'inference_speed', 'weight': 'False' },
        #  6: {'name':'gflops', 'weight': 'True' },
        #  7: {'name':'fps', 'weight': 'True' },
        #  8: {'name':'latency', 'weight': 'False' },
        #  9: {'name':'AssA', 'weight': 'True' },
}



markers = [
    "s", "D", "*", "h", "^", "<", ">", "1", "2", "3", "4", 
    "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", ".", ","]

class RecordInd:
    """
    Record data in batch
    """
    def __init__(self, batch_size=100, output_file="pareto_data.csv", columns=['id', 'generation', 'fitness', 'all_fitness']):
        self.batch_size = batch_size
        self.output_file = output_file
        self.batches = []
        self.columns = columns

        # Initialize the CSV file with a header
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.columns)

    def record(self, row):
        """Stop the timer for a specific function and update its statistics."""

        self.batches.append(row)

        # Save batch if it reaches the threshold
        if len(self.batches) >= self.batch_size:
            self.save_batch( )

    def save_batch(self):
        """Save the current batch of timings for a specific function to the CSV file and reset."""
        with open(self.output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            for row in self.batches:
                writer.writerow(row)
        self.batches = []  # Clear batch after saving


def stepify_pareto_points_2d(x, y, metric_directions):
    """
    Returns the pareto front points, including the steps, from the x/y points of a 2D pareto front
    Takes in the optimization directions for the x and y parameters as a list of booleans (True=Minimized, False=Maximized)
    """

    is_x_minimized, is_y_minimimized = metric_directions
    is_x_minimized = eval(is_x_minimized)
    is_y_minimimized = eval(is_y_minimimized)
    # sort for pareto steps
    y_argsort = np.argsort(y)
    x = x[y_argsort]
    y = y[y_argsort]
    x_argsort = np.argsort(x)
    x = x[x_argsort]
    y = y[x_argsort]

    last_x, last_y = x[0], y[0]
    x_steps = [last_x]
    y_steps = [last_y]
    # step direction is based on the optimization direction of each axis
    for i, x_val in enumerate(x):
        y_val = y[i]
        if last_x != x_val:
            # add the stair step
            if is_x_minimized:
                y_steps.append(last_y)
                x_steps.append(x_val)
            else:
                y_steps.append(y_val)
                x_steps.append(last_x)
            # add the point
            y_steps.append(y_val)
            x_steps.append(x_val)
        last_x, last_y = x_val, y_val
    return x_steps, y_steps

pareto_frames_output = os.path.join('.', "pareto_frames")
os.makedirs(pareto_frames_output, exist_ok=True)


def get_stat_from_pkl(folder_name='/home/hice1/yzhang3942/scratch/llm-guided-evolution/pr_run-6-25-25/checkpoints'):  #og: first_test
    """
    Get total number of individuals and valid inds from GE database
    """
    database_files = glob.glob(os.path.join(folder_name, '*.pkl'))
    checkpoint, start_gen = load_checkpoint(folder_name=folder_name, checkpoint_file=f"checkpoint_gen_{len(database_files)-1}.pkl")
    populations_list = []
    for i in range(len(database_files)):
        # collect fitness for each generation
        checkpoint, start_gen = load_checkpoint(folder_name=folder_name, checkpoint_file=f"checkpoint_gen_{i}.pkl")
        for item in checkpoint['population']:
            populations_list.extend(item)
        

    print(i, len(checkpoint['GLOBAL_DATA']), len(checkpoint['population']), len(set(populations_list)))
    
    df = pd.DataFrame(checkpoint['GLOBAL_DATA']).T
    finite_inds = sum(df[df.status == "completed"].fitness.apply(lambda x: np.isfinite(x[0])))
    print("total_inds", len(set(populations_list)), "total_valid_ind", finite_inds)

def collect_scores(folder_name='/home/hice1/yzhang3942/scratch/llm-guided-evolution/pr_run-6-25-25/checkpoints'): #OG: first_test
    """
    collect ind score from GE database
    """
    database_files = glob.glob(os.path.join(folder_name, '*.pkl'))
    
    scores_dict = dict()
    for i in range(len(database_files)):
        # collect fitness for each generation
        checkpoint, start_gen = load_checkpoint(folder_name=folder_name, checkpoint_file=f"checkpoint_gen_{i}.pkl")
        scores = []
        inds = []
        for item in  checkpoint['GLOBAL_DATA']:
            fitness = checkpoint['GLOBAL_DATA'][item]['fitness']
            print(fitness)
            if fitness is not None:
                if np.isfinite(sum(fitness)):
                    scores.append(fitness)
                    inds.append(item)
        scores = np.array(scores)
        scores_dict[i] = {'scores': scores, 'inds': inds}
    return scores_dict

def get_min_max(scores_dict: dict):
    """
    Get maxinum and minimum from scores
    """
    scores = np.vstack([scores_dict[i]['scores'] for i in range(len(scores_dict))])
    mins = scores.min(axis=0)
    maxs = scores.max(axis=0)
    return mins, maxs

def plot_1d_pareto_front(scores_dict, objectives=[1], maximize_ojb=[], fig=None, plot_flag=True, legend='mixtral'):
    
    name_objs = '_'.join([map_obj[item]['name'] for item in objectives])
    record = RecordInd(output_file=f"1d_pareto_data_{name_objs}.csv")
    
    mins, maxs = get_min_max(scores_dict)
    weight = eval(map_obj[objectives[0]]['weight'])
    ops = np.max if weight else np.min

    y_min = mins[objectives[0]] * 0.9
    y_max = maxs[objectives[0]] * 1.1

    y_scores = []
    optimal_score = None
    for i in range(len(scores_dict)):
        scores = scores_dict[i]['scores']
        inds = scores_dict[i]['inds']
        ones_column = np.array(range(len(scores))).reshape(-1, 1)
        scores = np.hstack((scores, ones_column))
        nondominated_scores = np.array(eps_sort(scores, objectives=objectives, maximize=maximize_ojb))
        if i == 0:
            optimal_score = nondominated_scores[0, objectives[-1]]
        else:
            optimal_score = ops([optimal_score, nondominated_scores[0, objectives[-1]]])
        y_scores.append(optimal_score)
    if fig is None:
        fig = plt.figure()
    # for item in baselines:
    #     values = baselines[item][objectives[-1]]
    #     plt.plot(range(len(scores_dict)), [values]*(len(scores_dict)), '--.', label=item)
    plt.plot(range(len(scores_dict)), np.log(y_scores), '-o', label=legend)
    plt.ylabel(map_obj[objectives[0]]['name'])
    plt.xlabel("Generation")
    plt.legend()
    plt.grid()
    if plot_flag:
        names = map_obj[objectives[0]]['name']
        plt.savefig(f'1d_{names}.png')
    return fig
    

def plot_2d_pareto_front(scores_dict, objectives=[0,1], maximize_ojb=[1], model='mixtral'):
    
    name_objs = '_'.join([map_obj[item]['name'] for item in objectives])
    record = RecordInd(output_file=f"{model}_2d_pareto_data_{name_objs}.csv")
    
    mins, maxs = get_min_max(scores_dict)

    x_min = mins[objectives[0]] * 0.9
    y_min = mins[objectives[1]] * 0.9
    x_max = maxs[objectives[0]] * 1.1
    y_max = maxs[objectives[1]] * 1.1

    for i in range(len(scores_dict)):
        scores = scores_dict[i]['scores']
        inds = scores_dict[i]['inds']
        ones_column = np.array(range(len(scores))).reshape(-1, 1)
        scores = np.hstack((scores, ones_column))
        nondominated_scores = np.array(eps_sort(scores, objectives=objectives, maximize=maximize_ojb))
        #import pdb; pdb.set_trace()
        print(f"Gen {i} Pareto inds \n")
        for j in nondominated_scores[:, -1]:
            print(inds[int(j)], scores[int(j), objectives])
            record.record([inds[int(j)], i, scores[int(j), objectives], scores[int(j), :]])
        print(f"==================== \n")
        #x = np.array(scores[gen_num][metrics_of_interest[0]])
        #y = np.array(scores[gen_num][metrics_of_interest[1]])
        x = nondominated_scores[:, objectives[0]]
        y = nondominated_scores[:, objectives[1]]
        metric_directions=[]
        for obj in objectives:
            metric_directions.append(map_obj[obj]['weight'])
        x_steps, y_steps = stepify_pareto_points_2d(x, y, metric_directions)

        plt.title(f"Pareto Front Generation {i:02d} - size {len(x)}")
        plt.scatter(scores[:, objectives[0]], scores[:, objectives[1]], alpha=0.5)
        #plt.scatter(scores[:, 0], scores[:, 1] *4 / 1024**2)
        plt.plot(x_steps, y_steps, label="Pareto Frontier", color="xkcd:blue")
        # for marker_id, item in enumerate(baselines):
        #     values = baselines[item]
        #     plt.plot(values[objectives[0]], values[objectives[1]], marker=markers[marker_id], label=item)
            #plt.text(0.5, 0, f'gen-{i}', fontsize=24, color='red')
        plt.legend(loc='lower right')
        plt.ylabel(map_obj[objectives[1]]['name'])
        plt.xlabel(map_obj[objectives[0]]['name'])
        plt.tight_layout()
        plt.grid()
        plt.ylim([y_min, y_max])
        plt.xlim([x_min, x_max])

        
        plt.savefig(f'{pareto_frames_output}/{model}_{name_objs}_test_{i}.png')
        plt.close()
    record.save_batch()

    frame_files = [os.path.join(pareto_frames_output, f"{model}_{name_objs}_test_{i}.png") for i in range(len(scores_dict))]
    # Open images
    frames = [Image.open(f) for f in frame_files]

    # Save as GIF
    gif_path = os.path.join('.', f'{model}_local_pareto_{name_objs}.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)



def plot_4d_pareto_front(scores_dict, objectives=[1,2], maximize_ojb=[2], model_name='Llama3.3-70b', file_name='pareto_fronts.pkl'):
    
    name_objs = '_'.join([map_obj[item]['name'] for item in objectives])
    output_file = f"{model_name}_4d_pareto_data_{name_objs}.csv"
    record = RecordInd(output_file=output_file)
    
    all_scores = []
    for i in range(len(scores_dict)):
        scores = scores_dict[i]['scores']
        all_scores.append(scores)
        inds = scores_dict[i]['inds']
        ones_column = np.array(range(len(scores))).reshape(-1, 1)
        scores = np.hstack((scores, ones_column))
        nondominated_scores = np.array(eps_sort(scores, objectives=objectives, maximize=maximize_ojb))
        #import pdb; pdb.set_trace()
        print(f"Gen {i} Pareto inds \n")
        for j in nondominated_scores[:, -1]:
            print(inds[int(j)], scores[int(j), objectives])
            record.record([inds[int(j)], i, scores[int(j), objectives]])
        print(f"==================== \n")
    
    with open(file_name, 'wb') as file:
        pickle.dump(all_scores, file)
    record.save_batch()
    df = pd.read_csv(output_file)
    df_sorted = df.sort_values(by='generation', ascending=False).drop_duplicates(subset='id')
    df_sorted.to_csv(f"{model_name}_4d_pareto_data_{name_objs}_unique.csv")
    return df

markers = [
    "v", "s", "D", "*", "h", "^", "<", ">", "1", "2", "3", "4", 
    "p", "*", "h", "H", "+", "x", "D", "d", "|", "_", ".", ","]

if __name__=='__main__':
    scores_dict_last = collect_scores(folder_name='first_test')
    df_sorted4 = plot_4d_pareto_front(scores_dict_last, objectives=[0, 1], maximize_ojb=[0], model_name='Llama3.3-70B')
    #scores_dict_last
    #maximize_ojb=[2, 3, 4, 5, 6]

    count = 0
    for model_name, df in zip([ 'Llama-3.3 GE + Optima'],  [df_sorted4]):
         item = df.groupby('generation').count().id.values
         plt.plot(range(len(item)), item, marker=markers[count], label=model_name)
         count +=1
    plt.legend()
    plt.grid()
    plt.xlabel('Generation')
    plt.ylabel('Number of Individuals in a Pareto Front')
    plt.savefig('Pareto_front_count.png')