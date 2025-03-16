import warnings
import os
import argparse
from synthetic_data_gen.data_generation import SyntheticImageGenerator, BlendingMode
import logging
logger = logging.getLogger(__name__)


def main(args=None):
    parser = argparse.ArgumentParser(description='Synthetic Image Generator')
    parser.add_argument('-in_dir', '--input_dir', type=str, required=True,
                        help='Path to the input directory. It must contain a backgrounds directory and a foregrounds directory')
    parser.add_argument('-out_dir', '--output_dir', type=str, required=True,
                        help='The directory where images and label files will be placed')
    parser.add_argument('--augmentation_path', type=str, default='synthetic_data_gen/transform.yml',
                        help='Path to albumentations augmentation pipeline file')
    parser.add_argument('-img_number', '--image_number', type=int,
                        required=True, help='Number of images to create')
    parser.add_argument('--max_objects_per_image', type=int,
                        default=3, help='Maximum number of objects per images')
    parser.add_argument('-i_w', '--image_width', type=int,
                        default=640, help='Width of the output images')
    parser.add_argument('-i_h', '--image_height', type=int,
                        default=480, help='Height of the output images')
    parser.add_argument('--fixed_image_sizes', default=True, action='store_false',
                        help='Whether or not to use fixed image sizes (default=true) => if false, the size of the background image is used and the height and width are ignored')
    parser.add_argument('--scale_foreground_by_background_size', default=True, action='store_false',
                        help='Whether the foreground images should be scaled based on the background size (default=true)')
    parser.add_argument('-s', '--scaling_factors', type=float, nargs=2, default=(0.25, 0.5),
                        help='Min and Max percentage size of the short side of the background image')
    parser.add_argument('--avoid_collisions', default=True, action='store_false',
                        help='Whether or not to avoid collisions (default=true)')
    parser.add_argument('--parallelize', default=False, action='store_true',
                        help='Whether or not to use multiple cores (default=false)')
    parser.add_argument('--yolo_input', default=False, action='store_true',
                        help='Has your background images been annotated in YOLO format?')
    parser.add_argument('-yolo', '--yolo_output', default=False, action='store_true',
                        help='Do you want to output the images in YOLO format?')
    parser.add_argument('-c', '--color_harmonization', default=False, action='store_true',
                        help='Do you want to apply color harmonization?')
    parser.add_argument('-c_a', '--color_harmon_alpha', type=float, default=0.5,
                        help='Color harmonization blending factor (0.0 to 1.0)')
    parser.add_argument('-c_rand', '--random_color_harmon_alpha', default=False, action='store_true',
                        help='Randomize the color harmonization blending factor (overrides --color_harmon_alpha)')
    parser.add_argument('--gaussian_options', type=int, nargs=2, default=(15, 30),
                        help='Kernel size and sigma for Gaussian blur blending mode')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Shows the images with the yolo labels (only works with yolo_output)')
    parser.add_argument('--blending_methods', type=str, nargs='+', default=['alpha', 'gaussian', 'poisson_normal', 'poisson_mixed', 'pyramid'],
                        help='Blending methods to use (alpha, gaussian, poisson_normal, poisson_mixed, pyramid). List of strings')
    parser.add_argument('-p_l', '--pyramid_blending_levels', type=int, default=6,
                        help='Number of levels for the pyramid blending method')
    parser.add_argument('--distractor_objects', type=str, nargs='+', default=[],
                        help='List of foreground images, which should be used as distractor objects')
    args = parser.parse_args()

    blending_methods = []
    for blending_method in args.blending_methods:
        if blending_method == 'alpha':
            blending_methods.append(BlendingMode.ALPHA_BLENDING)
        elif blending_method == 'gaussian':
            blending_methods.append(BlendingMode.GAUSSIAN_BLUR)
        elif blending_method == 'poisson_normal':
            blending_methods.append(BlendingMode.POISSON_BLENDING_NORMAL)
        elif blending_method == 'poisson_mixed':
            blending_methods.append(BlendingMode.POISSON_BLENDING_MIXED)
        elif blending_method == 'pyramid':
            blending_methods.append(BlendingMode.PYRAMID_BLEND)
        else:
            print(f'Blending method {blending_method} not found')
            import sys
            sys.exit(1)

    if args.fixed_image_sizes and args.yolo_input:
        warnings.warn(
            'Fixed image sizes is set to true, but yolo input is enabled. This is currently not supported!')
        quit()

    # set the logging level
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    if (args.random_color_harmon_alpha or args.color_harmon_alpha) and not args.color_harmonization:
        warnings.warn(
            'Color harmonization alpha is set (either random or specific), but color harmonization is not enabled. Ignoring the alpha value')

    # Divide the number of images by the number of blending methods
    img_number = int(args.image_number / len(blending_methods))
    logging.info(
        f'Generating {args.image_number} images with {img_number} images per blending method')

    # remove the output dir if it exists
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    else:
        os.makedirs(args.output_dir)
    data_generator = SyntheticImageGenerator(args.input_dir, args.output_dir, img_number, args.max_objects_per_image, args.image_width,
                                             args.image_height, args.fixed_image_sizes, args.augmentation_path, args.scale_foreground_by_background_size, args.scaling_factors,
                                             args.avoid_collisions, args.parallelize, args.yolo_input, args.yolo_output, args.color_harmonization, args.color_harmon_alpha,
                                             args.random_color_harmon_alpha, args.gaussian_options, args.debug, blending_methods, args.pyramid_blending_levels, args.distractor_objects)
    data_generator.generate_images()
