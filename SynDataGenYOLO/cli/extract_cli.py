import argparse
from SynDataGenYOLO.extract import extract_objects_from_labelme_data


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Extract objects from data labeled with LabelMe.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input images and labels.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path where output images will be saved.')
    parser.add_argument('--margin', type=int, default=10,
                        help='Margin (in pixels) to include around cropped objects.')
    if args is None:
        args = parser.parse_args()  # Parse args if called standalone
    else:
        args = parser.parse_args(args)  # Parse args when called from main CLI

    extract_objects_from_labelme_data(
        args.input_dir, args.output_dir, margin=args.margin)


if __name__ == "__main__":
    main()
