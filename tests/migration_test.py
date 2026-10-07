import os
import pickle
import pytest

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

def _validate_checkpoint_data(data, label):
    if data is None:
        raise ValueError(f"{label} checkpoint was not loaded")
    missing_keys = {"GLOBAL_DATA", "population"} - data.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"{label} checkpoint is missing required key(s): {missing}")

def _population_entry(population, individual):
    wrapped_entry = [individual]
    if wrapped_entry in population:
        return wrapped_entry
    if individual in population:
        return individual
    return wrapped_entry

def migrate_individuals(source_data, target_data, top_n=3):
    _validate_checkpoint_data(source_data, "source")
    _validate_checkpoint_data(target_data, "target")

    source_fitness = {
        key: value['fitness'][0]
        for key, value in source_data['GLOBAL_DATA'].items()
        if value.get('fitness')
    }
    
    sorted_source = sorted(source_fitness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for source_individual, _ in sorted_source:
        print(f"Migrating {source_individual} from source to target")
        target_data['GLOBAL_DATA'][source_individual] = source_data['GLOBAL_DATA'][source_individual]

        population_entry = _population_entry(source_data['population'], source_individual)
        if population_entry not in target_data['population']:
            target_data['population'].append(population_entry)
        
        del source_data['GLOBAL_DATA'][source_individual]
        if population_entry in source_data['population']:
            source_data['population'].remove(population_entry)

    return target_data
    
    
def save_checkpoint(checkpoint, gen, folder_name="checkpoints"):
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f'checkpoint_gen_{gen}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint, file)
    print(f"Checkpoint saved as {filename}")

def test_migrate_individuals_moves_top_fitness_entries():
    source = {
        "GLOBAL_DATA": {
            "a": {"fitness": (0.1,)},
            "b": {"fitness": (0.9,)},
            "c": {"fitness": (0.4,)},
        },
        "population": [["a"], ["b"], ["c"]],
    }
    target = {
        "GLOBAL_DATA": {},
        "population": [],
    }

    updated_target_data = migrate_individuals(source, target, top_n=2)

    assert set(updated_target_data["GLOBAL_DATA"]) == {"b", "c"}
    assert updated_target_data["population"] == [["b"], ["c"]]
    assert set(source["GLOBAL_DATA"]) == {"a"}
    assert source["population"] == [["a"]]

def test_migrate_individuals_requires_loaded_checkpoints():
    with pytest.raises(ValueError, match="source checkpoint was not loaded"):
        migrate_individuals(None, {"GLOBAL_DATA": {}, "population": []})

if __name__ == "__main__":
    checkpoint1, start_gen1 = loadcheckpoint(folder_name="checkpoints/island_1", checkpoint_file="checkpoint_gen_0.pkl")
    checkpoint2, start_gen2 = loadcheckpoint(folder_name="checkpoints/island_2", checkpoint_file="checkpoint_gen_0.pkl")

    updated_target_data = migrate_individuals(checkpoint1, checkpoint2)

