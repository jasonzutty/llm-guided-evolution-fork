import os
import pickle

def loadcheckpoint(folder_name="checkpoints", checkpoint_file=None):
    if not os.path.exists(folder_name):
        return None, None
    if checkpoint_file is None:
        checkpoint_files = sorted(os.listdir(folder_name), reverse=True)
        checkpoint_file = checkpoint_files[0] if checkpoint_files else None
    if checkpoint_file:
        filepath = os.path.join(folder_name, checkpoint_file)
        with open(filepath, 'rb') as file:
            checkpoint_data = pickle.load(file)
        print(f"Loaded checkpoint from {filepath}")
        start_gen = int(checkpoint_file.split('_')[2].split('.')[0])
        start_gen = start_gen + 1
        return checkpoint_data, start_gen
    return None, None

def migrate_individuals(source_data, target_data, top_n=3):
    source_fitness = {key: value['fitness'][0] for key, value in source_data['GLOBAL_DATA'].items() if value['fitness']}
    
    sorted_source = sorted(source_fitness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for source_individual, _ in sorted_source:
        print(f"Migrating {source_individual} from source to target")
        target_data['GLOBAL_DATA'][source_individual] = source_data['GLOBAL_DATA'][source_individual]

        target_data['population'].append(source_individual)
        
        del source_data['GLOBAL_DATA'][source_individual]
        source_data['population'].remove([source_individual])
    
    
def save_checkpoint(checkpoint, gen, folder_name="checkpoints"):
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f'checkpoint_gen_{gen}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint, file)
    print(f"Checkpoint saved as {filename}")

checkpoint1, start_gen1 = loadcheckpoint(folder_name="checkpoints/island_1", checkpoint_file="checkpoint_gen_0.pkl")
checkpoint2, start_gen2 = loadcheckpoint(folder_name="checkpoints/island_2", checkpoint_file="checkpoint_gen_0.pkl")

updated_target_data = migrate_individuals(checkpoint1, checkpoint2)


