
import argparse
from SynDataGenYOLO.show_images_with_bboxes import show_images_with_bboxes


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Show image or folder of images with ground truth and predicted bounding boxes')
    parser.add_argument('--input', type=str, required=True,
                        help='Either a directory containing images and labels or a single image')
    # parser.add_argument('--gt_labels', type=str, required=True,
    #                     help='Directory containing ground truth label files')
    parser.add_argument('--pred_labels', type=str, required=False,
                        help='Directory containing predicted label files')
    parser.add_argument('--write', type=bool, default=False,
                        help='Write the images with bounding boxes to a folder')
    parser.add_argument('--output', type=str, default='compare_bboxes_output',
                        help='Output folder to save the images with bounding boxes')
    parser.add_argument('--amount', type=int, default=10,
                        help='Amount of images to show (-1 for all)')
    parser.add_argument('--only_gt', type=bool, default=False,
                        help='Show only ground truth bounding boxes')
    parser.add_argument('--only_pred', type=bool, default=False,
                        help='Show only predicted bounding boxes')
    parser.add_argument('--classess', nargs='+', default=[],
                        help='Classes to show')

    if args is None:
        args = parser.parse_args()  # Parse args if called standalone
    else:
        args = parser.parse_args(args)  # Parse args when called from main CLI

    show_images_with_bboxes(args.input, args.pred_labels, args.write,
                            args.output, args.amount, args.only_gt, args.only_pred, args.classess)


if __name__ == "__main__":
    main()
