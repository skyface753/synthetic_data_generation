import os
import yaml

def load_class_map(class_map_path):
    """
    Load class map from a YAML file or auto-discover from JSON files in the input directory.
    Returns a dictionary mapping class names to indices.
        Example YAML format:
        names:
            - class1
            - class2
            - class3
        Example Return:
        {
            'class1': 0,
            'class2': 1,
            'class3': 2
        }
    """
    if class_map_path and os.path.exists(class_map_path):
        print(f"Loading class map from '{class_map_path}'...")
        with open(class_map_path, 'r') as f:
            class_data = yaml.safe_load(f)
            # Assumes YAML has a 'names' list like in YOLO config
            if 'names' in class_data and isinstance(class_data['names'], list):
                class_names = class_data['names']
                class_to_idx = {name: i for i, name in enumerate(class_names)}
                print(f"Found {len(class_to_idx)} classes: {list(class_to_idx.keys())}")
                return class_to_idx
            else:
                raise ValueError("YAML file must contain a 'names' key with a list of class names.")
    else:
        raise FileNotFoundError(f"Class map file '{class_map_path}' not found.")