import os
import json
# get all .json files in the directory


def extract_to_yolo(input_dir, output_dir, find_labels):
    # ask the user if the output dir should be cleaned
    if os.path.exists(output_dir):
        answer = input(
            f"Directory {output_dir} already exists. Do you want to delete it? (y/n): ")
        if answer == 'y':
            os.system(f"rm -r {output_dir}")
        else:
            exit()
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))
    os.makedirs(os.path.join(output_dir, 'labels'))

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
            output_path_yolo_label = os.path.join(
                output_dir, 'labels', json_file.replace('.json', '.txt'))
            # output_path_json = os.path.join(output_dir, json_file)
            # output_path_image = os.path.join(output_dir, image_name)
            output_path_image = os.path.join(
                output_dir, 'images', image_name)
            labels = []
            for shape in json_data['shapes']:
                shape_type = shape['shape_type']
                if shape_type == 'rectangle':
                    label = shape['label']
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]
                    imageHeight = json_data['imageHeight']
                    imageWidth = json_data['imageWidth']
                    x_center = (x1 + x2) / 2 / imageWidth
                    y_center = (y1 + y2) / 2 / imageHeight
                    width = abs(x1 - x2) / imageWidth
                    height = abs(y1 - y2) / imageHeight
                    # get the index of the label
                    if label in find_labels:
                        label = find_labels.index(label)
                    else:
                        print(f"Label {label} not found in {find_labels}")
                        continue
                    labels.append(
                        f"{label} {x_center} {y_center} {width} {height}")
                if shape_type == 'polygon':
                    label = shape['label']
                    points = shape['points']
                    # go through all points and get the min and max
                    x1, y1 = points[0]
                    x2, y2 = points[0]
                    for x, y in points:
                        x1 = min(x1, x)
                        x2 = max(x2, x)
                        y1 = min(y1, y)
                        y2 = max(y2, y)
                    imageHeight = json_data['imageHeight']
                    imageWidth = json_data['imageWidth']
                    x_center = (x1 + x2) / 2 / imageWidth
                    y_center = (y1 + y2) / 2 / imageHeight
                    width = abs(x1 - x2) / imageWidth
                    height = abs(y1 - y2) / imageHeight
                    # get the index of the label
                    if label in find_labels:
                        label = find_labels.index(label)
                    else:
                        print(f"Label {label} not found in {find_labels}")
                        continue
                    labels.append(
                        f"{label} {x_center} {y_center} {width} {height}")
            # skip if no labels
            if len(labels) == 0:
                print(f"No labels found in {json_file}")
                continue
            with open(output_path_yolo_label, 'w') as f:
                f.write('\n'.join(labels))
            os.system(f"cp {image_path} {output_path_image}")

            # with open(output_path_json, 'w') as f:
            #     json.dump(json_data, f)
            # os.system(f"cp {image_path} {output_path_image}")
            print(f"Saved to {output_path_yolo_label}")
