import argparse

from syndatagenyolo.extract_to_yolo import extract_to_yolo


def main(args=None):
    parser = argparse.ArgumentParser(description="Extract labelme json files to yolo")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="input directory containing labelme json files and images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output directory to save the /images and /labels",
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="Optional path to a YAML file with class names."
    )

    if args is None:
        args = parser.parse_args()  # Parse args if called standalone
    else:
        args = parser.parse_args(args)  # Parse args when called from main CLI

    extract_to_yolo(args.input_dir, args.output_dir, args.labels)
