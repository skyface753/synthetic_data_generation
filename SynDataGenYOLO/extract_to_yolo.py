import os
import json
import shutil
import yaml
import argparse
from collections import defaultdict
from syndatagenyolo.utils.classes_yaml import load_class_map

def get_bounding_box(shape):
    """
    Calculates the bounding box for a given shape (rectangle or polygon).
    Returns (x1, y1, x2, y2) or None if shape type is unsupported.
    """
    shape_type = shape.get('shape_type')
    points = shape.get('points', [])

    if not points:
        return None

    if shape_type == 'rectangle':
        # For rectangles, points are [top_left, bottom_right]
        (x1, y1), (x2, y2) = points
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    
    elif shape_type == 'polygon':
        # For polygons, find the min/max of all points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    else:
        print(f"Warning: Unsupported shape type '{shape_type}' found. Skipping.")
        return None


def extract_to_yolo(input_dir, output_dir, class_map_path=None):
    """
    Converts LabelMe JSON annotations to YOLO format.

    Args:
        input_dir (str): Directory containing JSON files and corresponding images.
        output_dir (str): Directory to save the YOLO formatted dataset.
        class_map_path (str, optional): Path to a YAML file containing the class map.
                                         If not provided, labels will be auto-discovered.
    """
    # --- 1. Setup Directories ---
    if os.path.exists(output_dir):
        answer = input(
            f"Directory '{output_dir}' already exists. Do you want to delete and recreate it? (y/n): ")
        if answer.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("Operation cancelled by user.")
            exit()
    
    images_out_dir = os.path.join(output_dir, 'images')
    labels_out_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_out_dir)
    os.makedirs(labels_out_dir)

    # --- 2. Determine Class Mapping ---
    class_to_idx = load_class_map(class_map_path) if class_map_path else None
    if class_to_idx is None:
        print("No class map provided. Auto-discovering labels from JSON files...")
        all_labels = set()
        json_files_for_scan = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        for json_file in json_files_for_scan:
            with open(os.path.join(input_dir, json_file), 'r') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    all_labels.add(shape['label'])
        
        # Sort labels alphabetically for consistent mapping
        sorted_labels = sorted(list(all_labels))
        class_to_idx = {name: i for i, name in enumerate(sorted_labels)}

        if not class_to_idx:
            print("Error: No labels found in any JSON file. Cannot proceed.")
            exit()
        print(f"Discovered {len(class_to_idx)} classes: {sorted_labels}")

    # --- 3. Process Annotations ---
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_name = data.get('imagePath')
        if not image_name:
            print(f"Warning: 'imagePath' not found in '{json_file}'. Skipping.")
            continue

        image_path = os.path.join(input_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found. Skipping annotation file '{json_file}'.")
            continue
        
        print(f"Processing: {json_file}")

        img_height = data.get('imageHeight')
        img_width = data.get('imageWidth')

        yolo_labels = []
        for shape in data.get('shapes', []):
            label_name = shape.get('label')
            if label_name not in class_to_idx:
                print(f"Warning: Label '{label_name}' in '{json_file}' is not in the class map. Skipping this annotation.")
                continue

            bbox = get_bounding_box(shape)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            class_idx = class_to_idx[label_name]

            # Convert to YOLO format (normalized center_x, center_y, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_labels.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # --- 4. Save Output Files ---
        if yolo_labels:
            # Save YOLO label file
            label_filename = os.path.splitext(image_name)[0] + '.txt'
            output_label_path = os.path.join(labels_out_dir, label_filename)
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            # Copy image file
            output_image_path = os.path.join(images_out_dir, image_name)
            shutil.copy(image_path, output_image_path)
        else:
            print(f"Info: No valid labels found for '{json_file}'. No output generated for this file.")

    # --- 5. Create data.yaml for YOLO training ---
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    class_names_list = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])
    
    yaml_content = {
        'path': os.path.abspath(output_dir), # Root directory
        'train': 'images', # train images path relative to 'path'
        'val': 'images',   # val images path relative to 'path' (can be adjusted)
        'names': class_names_list
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print("\n---------------------------------")
    print("Conversion complete!")
    print(f"YOLO dataset created at: '{output_dir}'")
    print(f"A 'data.yaml' file has been created at: '{data_yaml_path}'")
    print("You can use this YAML file for training your YOLO model.")
    print("---------------------------------")


# if __name__ == '__main__':
#     # You will need to install PyYAML: pip install pyyaml
#     parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to YOLO format.")
#     parser.add_argument('input_dir', type=str, help="Directory containing JSON files and images.")
#     parser.add_argument('output_dir', type=str, help="Directory to save the YOLO formatted dataset.")
#     parser.add_argument('--labels', type=str, default=None, help="Optional path to a YAML file with class names.")
    
#     args = parser.parse_args()
    
#     extract_to_yolo(args.input_dir, args.output_dir, args.labels)