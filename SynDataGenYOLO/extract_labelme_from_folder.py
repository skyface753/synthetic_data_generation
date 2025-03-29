import os
import json
# get all .json files in the directory


def extract_labelme_from_folder(input_dir, output_dir):
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    for json_file in json_files:
        with open(os.path.join(input_dir, json_file), 'r') as f:
            json_data = json.load(f)
            image_name = json_data['imagePath']
            image_path = os.path.join(input_dir, image_name)
            print(f"Processing {json_file}")
            if os.path.exists(image_path):
                print(f"Found image {image_path}")
            else:
                print(f"Image {image_path} not found")
                continue
            output_path_json = os.path.join(output_dir, json_file)
            output_path_image = os.path.join(output_dir, image_name)
            with open(output_path_json, 'w') as f:
                json.dump(json_data, f)
            os.system(f"cp {image_path} {output_path_image}")
            print(f"Saved {output_path_json} and {output_path_image}")
