import os
import glob
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
from shapely.geometry import Polygon
import tqdm


def extract_objects_from_labelme_data(input_dir, output_dir, margin=10):
    """
    Extract objects from LabelMe JSON annotations and save them as cropped images with optional margin.

    :param input_dir: Path to input directory containing LabelMe JSON files and images
    :param output_dir: Path to save the cropped images
    :param margin: Margin (in pixels) to include around the cropped object
    """
    # Create output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)
    # check if the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    # Get path to all json files in the input directory
    labelme_json_paths = glob.glob(os.path.join(input_dir, "*.json"))

    label_counts = dict()

    # for json_file in labelme_json_paths:
    for json_file in tqdm.tqdm(labelme_json_paths):
        # Open json file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Load base64 image
        # check if a image exists, with same name as the json file
        IMG_EXT = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        image_path = None
        for ext in IMG_EXT:
            if os.path.exists(os.path.join(input_dir, os.path.splitext(os.path.basename(json_file))[0] + ext)):
                image_path = os.path.join(input_dir, os.path.splitext(
                    os.path.basename(json_file))[0] + ext)
                break
        if image_path is None:
            # use the image data in the json file
            im = Image.open(BytesIO(base64.b64decode(
                data['imageData']))).convert('RGBA')
            im_array = np.asarray(im)
        else:
            im = Image.open(image_path).convert('RGBA')
            im_array = np.asarray(im)

        original_name = os.path.splitext(os.path.basename(json_file))[0]

        # Loop through all the annotations
        for annotation in data['shapes']:
            label = annotation['label']

            if label not in label_counts:
                label_counts[label] = 0
                os.makedirs(os.path.join(output_dir, label), exist_ok=True)

            # Extract object from image
            mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
            ImageDraw.Draw(mask_im).polygon(
                tuple(map(tuple, annotation['points'])), outline=1, fill=1)
            mask = np.array(mask_im)

            # Assemble new image (uint8: 0-255)
            new_img_array = np.empty(im_array.shape, dtype='uint8')

            # Colors (three first columns, RGB)
            new_img_array[:, :, :3] = im_array[:, :, :3]

            # Transparency (4th column)
            new_img_array[:, :, 3] = mask * 255

            # Convert to image
            new_im = Image.fromarray(new_img_array, "RGBA")
            x_min, y_min, x_max, y_max = Polygon(annotation['points']).bounds

            # Expand the bounding box by the margin
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(im.width, x_max + margin)
            y_max = min(im.height, y_max + margin)

            # Crop and save the image
            new_im = new_im.crop((x_min, y_min, x_max, y_max))
            new_im.save(os.path.join(
                output_dir, label, f'{label_counts[label]}-{original_name}.png'), optimize=True)
            # Image.save(new_im, os.path.join(output_dir, label, f'{label_counts[label]}.png'), optimize=True)
            label_counts[label] += 1
