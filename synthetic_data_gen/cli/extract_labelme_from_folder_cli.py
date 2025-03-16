import argparse
from synthetic_data_gen.extract_labelme_from_folder import extract_labelme_from_folder


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Extract labelme json files with images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='input directory containing labelme json files and images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory to save the extracted json files with images')
    args = parser.parse_args()

    extract_labelme_from_folder(args.input_dir, args.output_dir)
