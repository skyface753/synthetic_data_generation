from pathlib import Path
import warnings
from typing import List
import logging
logger = logging.getLogger(__name__)


def validate_input_directory(input_dir: Path, yolo_input: bool,
                             distractor_objects: List[str] = None):

    # Check if directory exists
    assert input_dir.exists(
    ), f'input_dir does not exist: {input_dir}'

    # Check if directory contains a foregrounds and backgrounds directory
    for p in input_dir.glob('*'):
        if p.name == 'foregrounds':
            foregrounds_dir = p
        elif p.name == 'backgrounds':
            backgrounds_dir = p
        elif p.name == 'labels':
            if yolo_input:
                labels_dir = p

    assert foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
    assert backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'
    if yolo_input:
        assert labels_dir is not None, 'labels sub-directory was not found in the input_dir'

    foregrounds_dict, categories = _validate_and_process_foregrounds(foregrounds_dir, yolo_input=yolo_input,
                                                                     distractor_objects=distractor_objects)
    background_images = _validate_and_process_backgrounds(backgrounds_dir)
    if yolo_input:
        labels_dict, categories = _validate_and_process_labels(
            labels_dir, categories)
    else:
        labels_dict = dict()

    return foregrounds_dict, categories, background_images, labels_dict


def _validate_and_process_foregrounds(foregrounds_dir: Path, yolo_input: bool = False,
                                      distractor_objects: List[str] = None):
    foregrounds_dict = dict()
    categories = []

    for category in foregrounds_dir.glob('*'):
        # check if we have a directory
        if not category.is_dir():
            warnings.warn(
                f'File found in foregrounds directory, ignoring: {category}')
            continue
        # check if the category is a distractor object
        if category.name in distractor_objects:
            warnings.warn(
                f'Distractor object found in foregrounds directory, ignoring: {category}')
            warnings.warn(
                f'Distractor objects are not supported yet. Coming soon...')
            continue

        # Add images inside category folder to foregrounds dictionary
        foregrounds_dict[category.name] = list(category.glob('*.png'))
        categories.append(
            {'name': category.name, 'id': len(categories)})

    assert len(
        foregrounds_dict) > 0, f'No valid foreground images were found in directory: {foregrounds_dir} '

    return foregrounds_dict, categories


def _validate_and_process_backgrounds(backgrounds_dir: Path):
    background_images = []

    for ext in ('*.png', '*.jpg', '*jpeg', '*.JPG'):
        background_images.extend(backgrounds_dir.glob(ext))

    assert len(
        background_images) > 0, f'No valid background images were found in directory: {backgrounds_dir}'

    return background_images


def _validate_and_process_labels(labels_dir: Path, categories: List[dict]):
    # Check for corresponding label files
    labels_dict = dict()

    for label in labels_dir.glob('*'):
        if label.suffix == '.txt':
            labels_dict[label.stem] = label

    assert len(
        labels_dict) > 0, f'No valid label files were found in directory: {labels_dir}'

    # Check if the number of labels match the number of background images
    # assert len(background_images) == len(
    #     labels_dict), f'Number of label files does not match the number of background images'

    # when we have labels here, we need to update the categories from the foregrounds
    # so we have all labels in the categories (startet with the background categories, and then added the labels from the foregrounds)
    categories_from_labels = []
    for label in labels_dict.values():
        with open(label, 'r') as f:
            for line in f.readlines():
                category = line.split(' ')[0]
                if category not in categories_from_labels:
                    categories_from_labels.append(category)
    categories_from_labels.sort()
    # categories [{'name': 'Tardigrade', 'id': 0}]
    # categories_from_labels['0']
    for category in categories_from_labels:
        # check if the id is already in the categories
        found = False
        for cat in categories:
            if int(cat['id']) == int(category):
                found = True
                break
        if not found:
            categories.append(
                {'name': category, 'id': len(categories
                                             )})
    logger.debug(categories)

    return labels_dict, categories
