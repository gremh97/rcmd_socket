import os
import hashlib
import json


def calculate_file_hash(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def load_json_file(json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)


def load_previous_json(tasks, json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)
    else:
        return {task: {'dir':'', 'models':[]} for task in tasks}


def get_current_models(base_dir, tasks):
    task_models = {}
    for task in tasks:
        task_path = os.path.join(base_dir, task, 'input')
        if os.path.exists(task_path):
            # task_models[task] = []
            task_models[task] = {'dir':task_path, 'models':[]}
            model_files = os.listdir(task_path)
            for model_file in model_files:
                file_path = os.path.join(task_path, model_file)
                if os.path.isfile(file_path):
                    model_name, ext = os.path.splitext(model_file)
                    if ext == '.lne':
                        model_hash = calculate_file_hash(file_path)
                        task_models[task]['models'].append({'name': model_name, 'hash': model_hash})
        else:
          task_models[task] = {'dir': task_path, 'models': []}
    return task_models

def compare_models(tasks, previous_models, current_models):
    changes = {'added': {}, 'deleted': {}, 'modified': {}}

    for task in tasks:
        previous_models_dict    = {model['name']:model['hash'] for model in previous_models.get(task, {}).get('models', [])}   
        current_models_dict     = {model['name']:model['hash'] for model in current_models.get(task, {}).get('models', [])} 

        # Added models
        added_models = [model for model in current_models_dict.keys() if model not in previous_models_dict.keys()]
        changes['added'][task] = added_models

        # Deleted models
        deleted_models = [model for model in previous_models_dict.keys() if model not in current_models_dict.keys()]
        changes['deleted'][task] = deleted_models
        
        # Modified models
        modified_models = [model for model, hash in current_models_dict.items() if model in previous_models_dict.keys() and hash != previous_models_dict[model]]
        changes['modified'][task] = modified_models
    return changes


def save_json(json_file_path, current_models):
    with open(json_file_path, 'w') as json_file:
        json.dump(current_models, json_file, indent=4)


def update_model_list(  tasks = ['classify', 'detect', 'detect_yolo'],
                        base_dir = '/home/evaluate',
                        json_file_path = '/home/models_by_task.json'):

    tasks = tasks if isinstance(tasks, list) else [tasks]
    
    previous_models = load_previous_json(tasks, json_file_path)
    current_models = get_current_models(base_dir, tasks)
    print(json.dumps(current_models, indent=4))
    changes = compare_models(tasks, previous_models, current_models)

    # print changes
    print("Added Models:")
    for task in changes['added'].keys():
      print(f"\tTask: {task}, Model: {', '.join(map(str, changes['added'].get(task)))}")

    print("\nDeleted Models:")
    for task in changes['deleted'].keys():
      print(f"\tTask: {task}, Model: {', '.join(map(str, changes['deleted'].get(task)))}")

    print("\nModified Models:")
    for task in changes['modified'].keys():
      print(f"\tTask: {task}, Model: {', '.join(map(str, changes['modified'].get(task)))}")


    save_json(json_file_path, current_models)
    print(f"\nUpdated: {json_file_path}")
    return current_models


def find_task_by_model(model_name, json_file_path):
    data = load_json_file(json_file_path)
    for task, info in data.items():
        for model in info.get('models', []):
            if model['name'] == model_name:
                return task
    return None


if __name__ == '__main__':
   _ = update_model_list()
