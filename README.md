# Synthetic Data Generation

A bunch of scripts to generate synthetic images for YOLO.

## Install

1. Install the required packages:

```bash
pip install SynDataGenYOLO
```

## Tools

### Extract Labelme Objects

With this scripts, you can extract objects, which are annotated with labelme, from images.

```bash
SynDataGenYOLO extract --input_dir INPUT_DIR --output_dir OUTPUT_DIR --margin MARGIN
```

- `INPUT_DIR`: The directory where the images are stored.
- `OUTPUT_DIR`: The directory where the extracted objects will be stored.
- `MARGIN`: The margin around the object. Usefull to add some space around the object and blend it into the background.

### Generate Synthetic Images

With this script, you can generate synthetic images with the extracted objects and corresponding backgrounds.

Minimal example:

```bash
SynDataGenYOLO generate --input_dir INPUT_DIR --output_dir OUTPUT_DIR --image_number IMAGE_NUMBER
```

- `INPUT_DIR`: The directory where the extracted objects are stored.
- `OUTPUT_DIR`: The directory where the synthetic images will be stored.
- `IMAGE_NUMBER`: The number of synthetic images to generate.

##### INPUT_DIR

The input directory should have the following structure:

```
INPUT_DIR
├── backgrounds
│   ├── background_1.jpg
│   ├── background_2.jpg
│   └── ...
├── foregrounds
|   ├── object_1
|   │   ├── object_1_1.png
|   │   ├── object_1_2.png
|   │   └── ...
|   ├── object_2
|   │   ├── object_2_1.png
|   │   ├── object_2_2.png
|   │   └── ...
|   └── ...
└── labels (optional - use with '--yolo_input')
    ├── background_1.txt
    ├── background_2.txt
    └── ...
```

Maximal example:

```bash
SynDataGenYOLO generate --input_dir INPUT_DIR --output_dir OUTPUT_DIR --image_number IMAGE_NUMBER --augmentation_path AUGMENTATION_PATH --max_objects_per_image MAX_OBJECTS_PER_IMAGE --image_width IMAGE_WIDTH --image_height IMAGE_HEIGHT --fixed_image_sizes --scale_foreground_by_background_size --scaling_factors SCALING_FACTORS SCALING_FACTORS --avoid_collisions --parallelize --yolo_input --yolo --color_harmon_alpha COLOR_HARMON_ALPHA --color_harmon_random --gaussian_options GAUSSIAN_OPTIONS GAUSSIAN_OPTIONS --debug --blending_methods BLENDING_METHODS BLENDING_METHODS --pyramid_blending_levels PYRAMID_BLENDING_LEVELS --distractor_objects DISTRACTOR_OBJECTS DISTRACTOR_OBJECTS
```

- `AUGMENTATION_PATH`: Path to a albumentations augmentation file.
- `MAX_OBJECTS_PER_IMAGE`: The maximum number of objects per image.
- `IMAGE_WIDTH`: The width of the generated images.
- `IMAGE_HEIGHT`: The height of the generated images.
- `FIXED_IMAGE_SIZES`: If set, the images will have the same size.
- `SCALE_FOREGROUND_BY_BACKGROUND_SIZE`: If set, the foreground objects will be scaled by the background size.
- `SCALING_FACTORS`: The scaling factors for the foreground objects.
- `AVOID_COLLISIONS`: If set, the objects will be placed in a way that they do not overlap.
- `PARALLELIZE`: If set, the generation will be parallelized using multiple processes.
- `YOLO_INPUT`: If set, the background images can contain yolo annotations.
- `YOLO`: If set, the generated images will have yolo annotations. Else COCO annotations will be used.
- `COLOR_HARMON_ALPHA`: The alpha value for the color harmonization.
- `COLOR_HARMON_RANDOM`: If set, the color harmonization will be random.
- `GAUSSIAN_OPTIONS`: The gaussian options for the blending. `kernel_size` and `sigma` (e.g. 5 1).
- `DEBUG`: If set, the debug mode will be activated.
- `BLENDING_METHODS`: The blending methods for the foreground objects. See below.
- `PYRAMID_BLENDING_LEVELS`: The number of pyramid blending levels.
- `DISTRACTOR_OBJECTS`: The names of foreground objects which should be used as distractor objects. (Not implemented yet, but will exclude these objects from the annotation file.)

#### Blending Methods

The blending methods are defined as follows:

- 'ALPHA': Alpha blending.
- 'GAUSSIAN': Gaussian blending.
- 'PYRAMID': Pyramid blending.
- 'POISSON_NORMAL': Poisson blending with normal blending (using the `cv2.seamlessClone()` function).
- 'POISSON_MIXED': Poisson blending with mixed blending (using the `cv2.seamlessClone()` function).

The blending methods can be combined with a space.

### Mix Datasets

```bash
SynDataGenYOLO mix --input_dirs INPUT_DIRS --output_dir OUTPUT_DIR --output_splits OUTPUT_SPLITS --percent_sets PERCENT_SETS --test_dataset TEST_DATASET --fixed_data_path FIXED_DATA_PATH --class_names CLASS_NAMES
```

- `INPUT_DIRS`: The directories where the datasets are stored.
- `OUTPUT_DIR`: The directory where the mixed dataset will be stored.
- `OUTPUT_SPLITS`: The output splits for the mixed dataset. (eg. `0.8 0.2` => 80% train, 20% validation)
- `PERCENT_SETS`: The percentage of the datasets which should be used. (eg. `0.5 0.5` => 50% of each dataset)
- `TEST_DATASET`: The dataset which should be used as test dataset.
- `FIXED_DATA_PATH`: Use a absolute path in the `data.yaml` file.
- `CLASS_NAMES`: The class names for the mixed dataset. (eg. `class_1 class_2 class_3`)

## Contribute

1. Clone the repository:

```bash
git clone
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install the package in editable mode:

```bash
pip install -e .
```

## Publish

1. Update the version in `pyproject.toml`.

2. Update the `CHANGELOG.md`.

3. Build the package:

```bash
pip install --upgrade build
python -m build
```

4. Publish the package:

```bash
pip install --upgrade twine
python -m twine upload dist/*
```
