import os
import pandas as pd
import subprocess
import numpy as np


def write_dataset(df, output_dir_images, output_dir_labels):
    try:
        for index, row in df.iterrows():
            # print(row)
            dataset_dir = row['input_dir']
            # copy images
            image_name = str(row['df_index']) + "_" + row['images']
            label_name_out = str(row['df_index']) + "_" + row['labels']
            # os.system(
            #     f'cp {os.path.join(dataset_dir, "images", row["images"])} {os.path.join(output_dir_images, image_name)}')
            subprocess.run(['cp', os.path.join(dataset_dir, "images", row["images"]), os.path.join(
                output_dir_images, image_name)], check=True)
            # copy labels
            os.system(
                f'cp {os.path.join(dataset_dir, "labels", row["labels"])} {os.path.join(output_dir_labels, label_name_out)}')
            # print('Copied:', row['images'], 'to', image_name)
    except Exception as e:
        print('Error:', e)
        quit()


def merge_datasets_with_percentages(datasets, percentages, random_state=None):
    """
    Merge multiple datasets with specified percentages without duplicates,
    constrained by the smallest dataset.

    Parameters:
    - datasets: List of pandas DataFrames to merge
    - percentages: List of corresponding percentages (should sum to 100)
    - random_state: Seed for reproducibility

    Returns:
    - Merged DataFrame with specified percentage of each input dataset
    """
    # Validate inputs
    if len(datasets) != len(percentages):
        print(str(len(datasets)) + "!=" + str(len(percentages)))
        raise ValueError("Number of datasets must match number of percentages")

    if abs(sum(percentages) - 100) > 1e-10:
        raise ValueError("Percentages must sum to 100")
    # Find the size of the smallest dataset
    min_dataset_size = min(len(df) for df in datasets)
    print("Minimum dataset size:", min_dataset_size)
    # Calculate sample sizes based on the minimum dataset size
    merged_samples = []
    index = 0
    for df, percentage in zip(datasets, percentages):
        # Calculate sample size based on minimum dataset size and percentage
        sample_size = int(np.round(min_dataset_size * (percentage / 100)))
        print(f"Sample size for dataset {index}: {sample_size}")

        # Take a random sample without replacement
        sampled_df = df.sample(
            n=sample_size, replace=False, random_state=random_state)
        sampled_df['df_index'] = index
        index += 1
        merged_samples.append(sampled_df)

    # Combine the sampled datasets
    merged_dataset = pd.concat(merged_samples, ignore_index=True)
    return merged_dataset


def validate_percentages(original_datasets, merged_dataset, expected_percentages):
    # Verify percentage representation
    print("\nPercentage Verification:")
    for i, (df, expected_pct) in enumerate(zip(original_datasets, expected_percentages), 1):
        # Count how many rows from the original dataset are in the merged dataset
        rows_in_merged = len(merged_dataset[merged_dataset['df_index'] == i-1])
        actual_pct = (rows_in_merged / len(merged_dataset)) * 100
        print(f"Dataset {i}:")
        print(f"  Expected: {expected_pct}%")
        print(f"  Actual: {actual_pct:.2f}%")

        # More precise percentage checking
        assert abs(
            actual_pct - expected_pct) <= 2.5, f"Percentage for dataset {i} is not within acceptable range"

    print("\nAll checks passed successfully!")


def mix_datasets(input_dirs, test_dataset, percent_sets, output_splits, output_dir, fixed_data_path, class_names):

    if fixed_data_path == '':
        fixed_data_path = None

    if test_dataset is None:
        print('Error: please provide a test dataset')
        exit()

    if percent_sets is None:
        percent_sets = [1] * len(input_dirs)
    if len(input_dirs) != len(percent_sets):
        print('Error: the number of input directories and percentage sets must be the same')
        exit()

    # delete output directory if it exists
    if os.path.exists(output_dir):
        os.system(f'rm -r {output_dir}')

    # ceck for more than one input directory
    if len(input_dirs) >= 1:
        print('Merging datasets...')
        datasets = []
        for input_dir in input_dirs:
            # check if input directory exists
            if not os.path.exists(input_dir):
                print(f'Error: input directory {input_dir} does not exist')
                continue
            # check if input directory contains images and labels
            if not os.path.exists(os.path.join(input_dir, 'images')) or not os.path.exists(os.path.join(input_dir, 'labels')):
                print(
                    f'Error: input directory {input_dir} does not contain images and labels')
                continue
            df = pd.DataFrame()
            all_images = [f for f in os.listdir(os.path.join(input_dir, 'images'))
                          if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG') or f.endswith('.PNG')]
            all_labels = [f for f in os.listdir(os.path.join(input_dir, 'labels'))
                          if f.endswith('.txt')]
            # check if all images have labels
            for image in all_images:
                # label = image.replace('.jpg', '.txt')
                label = image.replace('.jpg', '.txt').replace(
                    '.png', '.txt').replace('.JPG', '.txt').replace('.PNG', '.txt')
                if label not in all_labels:
                    print(
                        f'Error: label file {label} is missing for image {image}')
                    continue
            # add images and labels to dataframe
            df['images'] = all_images
            # get the corresponding label
            df['labels'] = [image.replace('.jpg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt').replace('.PNG', '.txt')
                            for image in all_images]
            df['input_dir'] = input_dir
            datasets.append(df)
            # print('Dataset:', input_dir)
            # print(df)

        sampled_dataset = merge_datasets_with_percentages(
            datasets, percent_sets)

        validate_percentages(datasets, sampled_dataset, percent_sets)

        percentages = []
        for i, ds in enumerate(datasets):
            percentages.append(
                len(sampled_dataset[sampled_dataset['df_index'] == i]) / len(sampled_dataset))
        print('Percentages (after sampling):', percentages)
        # quit()

        # calc the train and val splits
        train_split = int(len(sampled_dataset) * output_splits[0])
        val_split = int(len(sampled_dataset) * output_splits[1])
        # read the test data
        test_df = pd.DataFrame()
        test_df['images'] = [f for f in os.listdir(os.path.join(test_dataset, 'images'))
                             if f.endswith('.jpg') or f.endswith('.png')]
        test_df['labels'] = [image.replace('.jpg', '.txt').replace('.png', '.txt')
                             for image in test_df['images']]
        test_df['input_dir'] = test_dataset
        test_df['df_index'] = len(datasets)
        test_size = len(test_df)
        print('Train split:', train_split)
        print('Val split:', val_split)
        print('Test size:', test_size)
        # print('Test dataset:')
        # print(test_df)
        # split the dataset
        train_dataset = sampled_dataset[:train_split]
        val_dataset = sampled_dataset[train_split:]

        # write to csv
        # train_dataset.to_csv('train_dataset.csv', index=False)
        # val_dataset.to_csv('val_dataset.csv', index=False)

        # create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create output directories
        output_images_dir_train = os.path.join(output_dir, 'train', 'images')
        output_labels_dir_train = os.path.join(output_dir, 'train', 'labels')
        output_images_dir_val = os.path.join(output_dir, 'val', 'images')
        output_labels_dir_val = os.path.join(output_dir, 'val', 'labels')
        output_images_dir_test = os.path.join(output_dir, 'test', 'images')
        output_labels_dir_test = os.path.join(output_dir, 'test', 'labels')
        if not os.path.exists(output_images_dir_train):
            os.makedirs(output_images_dir_train)
        if not os.path.exists(output_labels_dir_train):
            os.makedirs(output_labels_dir_train)
        if not os.path.exists(output_images_dir_val):
            os.makedirs(output_images_dir_val)
        if not os.path.exists(output_labels_dir_val):
            os.makedirs(output_labels_dir_val)
        if not os.path.exists(output_images_dir_test):
            os.makedirs(output_images_dir_test)
        if not os.path.exists(output_labels_dir_test):
            os.makedirs(output_labels_dir_test)

        # write the train set
        write_dataset(train_dataset, output_images_dir_train,
                      output_labels_dir_train)

        write_dataset(val_dataset, output_images_dir_val,
                      output_labels_dir_val)
        write_dataset(test_df, output_images_dir_test,
                      output_labels_dir_test)
        # get all classes
        classes = []
        # from the labels
        for label in os.listdir(output_labels_dir_train):
            with open(os.path.join(output_labels_dir_train, label), 'r') as f:
                for line in f:
                    classes.append(line.split()[0])
        classes = list(set(classes))
        # sort
        classes.sort()
        print('Classes:', classes)

        if len(classes) != len(class_names):
            print(
                'Error: the number of classes and class names must be the same')
            exit()

        # write a data.yaml file
        if fixed_data_path is not None:
            absolute_path = fixed_data_path
            if absolute_path[-1] == '/':
                absolute_path = absolute_path[:-1]
            absolute_path = absolute_path + '/' + output_dir
        else:
            absolute_path = os.path.abspath(output_dir)

        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            # absolute_path = os.path.abspath(output_dir)
            f.write('train: ' + os.path.join(absolute_path, 'train') + '\n')
            f.write('val: ' + os.path.join(absolute_path, 'val') + '\n')
            f.write('test: ' + os.path.join(absolute_path, 'test') + '\n')
            f.write('nc: ' + str(len(set(classes))) + '\n')
            f.write('names: ' + '\n')
            for class_name in class_names:
                f.write('  - ' + class_name + '\n')
        # write the number of images per split to a file (for debugging)
        with open(os.path.join(output_dir, 'num_images.txt'), 'w') as f:
            f.write('Train: ' + str(len(train_dataset)) + '\n')
            f.write('Val: ' + str(len(val_dataset)) + '\n')
            f.write('Test: ' + str(len(test_df)) + '\n')

        print('Done')
    else:
        print('Error: please provide one or more input directories')
