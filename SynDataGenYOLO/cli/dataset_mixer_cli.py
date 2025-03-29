import argparse
from SynDataGenYOLO.dataset_mixer import mix_datasets


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Extract images and labels from multiple yolo datasets and save them in a new directory')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                        help='input directories containing a yolo dataset (images and labels)')
    parser.add_argument('--test_dataset', type=str, required=False,
                        help='input directory containing a yolo dataset (/images and /labels) to be used as test set')
    parser.add_argument('--percent_sets', type=float, nargs='+', required=False,
                        help='percentage of the dataset to be taken from each input directory')
    parser.add_argument('--output_splits', type=int, required=False, nargs=2,
                        help='percentage of splits to create from the dataset (train and val)', default=[0.8, 0.2])
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory to save the mixed dataset')
    parser.add_argument('--fixed_data_path', type=str, required=False,
                        help='fixed data path to be used in the data.yaml file')
    parser.add_argument('--class_names', type=str, nargs='+',
                        help='class names to be used in the data.yaml file')

    if args is None:
        args = parser.parse_args()  # Parse args if called standalone
    else:
        args = parser.parse_args(args)  # Parse args when called from main CLI

    mix_datasets(args.input_dirs, args.test_dataset, args.percent_sets,
                 args.output_splits, args.output_dir, args.fixed_data_path, args.class_names)
