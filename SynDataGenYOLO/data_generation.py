from SynDataGenYOLO.utils.modes import BlendingMode, OutputMode
from SynDataGenYOLO.utils.blending import poisson_blend_rgba, pyramid_blend
from SynDataGenYOLO.utils.data_gen import validate_input_directory
import json
import warnings
from pathlib import Path
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import albumentations as A
from joblib import Parallel, delayed
from typing import List
import cv2
import PIL
import logging
import importlib.resources as pkg_resources
import SynDataGenYOLO.data
logger = logging.getLogger(__name__)


class SyntheticImageGenerator:
    def __init__(self, input_dir: str, output_dir: str, image_number: int, max_objects_per_image: int,
                 image_width: int, image_height: int, fixed_image_sizes: bool, augmentation_path: str, scale_foreground_by_background_size: bool,
                 scaling_factors: List[int], avoid_collisions: bool, parallelize: bool, yolo_input: bool, output_mode: OutputMode,
                 color_harmonization: bool, color_harmon_alpha: float, random_color_harmon_alpha: bool, gaussian_options: List[int], debug: bool,
                 blending_methods: List[str], pyramid_blending_levels: int, distractor_objects: List[str], overwrite_output: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_number = image_number
        self.max_objects_per_image = max_objects_per_image
        self.image_width = image_width
        self.image_height = image_height
        self.fixed_image_sizes = fixed_image_sizes
        self.zero_padding = 8
        # umwandeln in Path kommt in _validate_augmentation_path
        self.augmentation_path = augmentation_path
        self.scale_foreground_by_background_size = scale_foreground_by_background_size
        self.scaling_factors = scaling_factors
        self.avoid_collisions = avoid_collisions
        self.parallelize = parallelize
        self.yolo_input = yolo_input
        self.output_mode = output_mode
        self.categories = []
        self.color_harmonization = color_harmonization
        self.color_harmon_alpha = color_harmon_alpha
        self.random_color_harmon_alpha = random_color_harmon_alpha

        self.gaussian_options = gaussian_options
        self.debug = debug

        self.blending_methods = blending_methods
        self.pyramid_blending_levels = pyramid_blending_levels
        self.distractor_objects = distractor_objects
        self.overwrite_output = overwrite_output

        if self.output_mode == OutputMode.CLASSIFICATION_SINGLE and self.yolo_input:
            raise ValueError(
                'Output mode CLASSIFICATION_SINGLE is not supported with YOLO input')
        if self.output_mode == OutputMode.CLASSIFICATION_SINGLE and self.max_objects_per_image > 1:
            raise ValueError(
                'Output mode CLASSIFICATION_SINGLE is not supported with multiple objects per image')

        self.foregrounds_dict, self.categories, self.background_images, self.labels_dict = validate_input_directory(
            self.input_dir, self.yolo_input, self.distractor_objects)
        # self._validate_output_directory()
        self._make_output_dir()
        self._validate_augmentation_path()

    # def _validate_output_directory(self):
    #     # Check if directory is empty
    #     assert len(list(self.output_dir.glob('*'))
    #                ) == 0, f'output_dir is not empty: {self.output_dir}'

    #     # Create output directory
    #     self.output_dir.mkdir(exist_ok=True)

    def _validate_augmentation_path(self):
        # Check if augmentation pipeline file exists
        # check if False
        print(
            f'augmentation_path: {self.augmentation_path} is {type(self.augmentation_path)}')
        if self.augmentation_path is False or self.augmentation_path == 'False' or self.augmentation_path == 'false':
            self.augmentation_path = None
            return
        if self.augmentation_path is None or self.augmentation_path == '':
            # Load default from package
            with pkg_resources.path(SynDataGenYOLO.data, 'default_augmentation.yml') as default_augmentation_path:
                self.augmentation_path = default_augmentation_path
            print(f'using default augmentation path: {self.augmentation_path}')
        self.augmentation_path = Path(self.augmentation_path)
        if self.augmentation_path.is_file() and self.augmentation_path.suffix == '.yml':
            self.transforms = A.load(
                self.augmentation_path, data_format='yaml')
        else:
            self.transforms = None
            raise FileNotFoundError(
                f'{self.augmentation_path} is not a file. No augmentations will be applied')

    def _generate_image(self, image_number: int):
        # Randomly choose a background image
        background_image_path = random.choice(self.background_images)

        # print(
        #     f'Generating image {image_number} with background {background_image_path}')
        num_foreground_images = random.randint(1, self.max_objects_per_image)
        foregrounds = []
        for i in range(num_foreground_images):
            # Randomly choose a foreground
            category = random.choice(list(self.foregrounds_dict.keys()))
            foreground_path = random.choice(self.foregrounds_dict[category])

            foregrounds.append({
                'category': category,
                'image_path': foreground_path
            })

        # Generate the composed foregrounds and background
        fg_images, mask_images, composite, annotations, foreground_positions, fg_image_sizes, fg_images_sized, mask_images_sized = self._gen_compose_images(
            foregrounds, background_image_path)

        # Paste foregrounds onto background with the blending modes
        blending_mode_images = self._paste_foregrounds(
            fg_images, mask_images, composite, foreground_positions, fg_image_sizes, fg_images_sized, mask_images_sized)

        for blending_mode_image in blending_mode_images:
            save_filename = f'{image_number:0{self.zero_padding}}_{blending_mode_image["blending_mode"]}'
            if self.output_mode == OutputMode.CLASSIFICATION_SINGLE:
                # get the class of the first foreground
                class_name = foregrounds[0]['category']
                save_filename = f'{class_name}/{save_filename}'

                output_path = self.output_dir / f'{save_filename}.jpg'
                print(f'writing {output_path}')
            elif self.output_mode == OutputMode.COCO:
                output_path = self.output_dir / f'{save_filename}.jpg'
            elif self.output_mode == OutputMode.YOLO:
                output_path = self.output_dir / f'images/{save_filename}.jpg'

            blending_mode_image['image'] = blending_mode_image['image'].convert(
                'RGB')
            blending_mode_image['image'].save(
                output_path, optimize=True, quality=70)

            annotations['imagePath'] = f'{save_filename}.jpg'
            if self.output_mode == OutputMode.COCO:
                annotations_output_path = self.output_dir / \
                    f'{save_filename}.json'
                with open(annotations_output_path, 'w+') as output_file:
                    json.dump(annotations, output_file)
            elif self.output_mode == OutputMode.YOLO:
                if self.yolo_input:
                    # get the corresponding label file
                    # check if the label file exists
                    if background_image_path.stem in self.labels_dict:
                        old_label_lines = []
                        old_label_file = self.labels_dict[background_image_path.stem]
                        # read the label file
                        with open(old_label_file, 'r') as f:
                            old_label_lines = f.readlines()
                    else:
                        old_label_lines = None  # no old label file
                        logger.debug(
                            f'Label file for {background_image_path.stem} not found, skipping...')
                # create the new annotations
                new_label_lines = []
                for shape in annotations['shapes']:
                    x_center, y_center, width, height = self._shape_to_yolo(
                        shape, annotations['imageWidth'], annotations['imageHeight'])
                    # get the id of the category
                    category_id = None
                    for category in self.categories:
                        if category['name'] == shape['label']:
                            category_id = category
                            break
                    if category_id is None:
                        warnings.warn(
                            f'category {shape["label"]} not found in categories')
                        continue
                    new_label_lines.append(
                        f'{category_id["id"]} {x_center} {y_center} {width} {height}\n')

                # Save label file
                label_output_path = self.output_dir / \
                    f'labels/{save_filename}.txt'
                with open(label_output_path, 'w+') as output_file:
                    if self.yolo_input and old_label_lines:
                        for line in old_label_lines:
                            output_file.write(line)
                        # newline
                        output_file.write('\n')
                    for line in new_label_lines:
                        output_file.write(line)
                # write a classes file
                classes_output_path = self.output_dir / 'classes.txt'
                # order the categories by id
                self.categories.sort(key=lambda x: x['id'])
                with open(classes_output_path, 'w+') as output_file:
                    for category in self.categories:
                        output_file.write(f'{category["name"]}\n')
            # print(f'Generated image: {output_path}')
            logger.debug(f'Generated image: {output_path}')
        return

    def _shape_to_yolo(self, shape, image_width, image_height):
        # Get the x (center), y(center), width, and height of the bounding box
        leftest_x = min([point[0] for point in shape['points']])
        rightest_x = max([point[0] for point in shape['points']])
        top_y = min([point[1] for point in shape['points']])
        bottom_y = max([point[1] for point in shape['points']])
        x_center = (leftest_x + rightest_x) / 2
        y_center = (top_y + bottom_y) / 2
        width = rightest_x - leftest_x
        height = bottom_y - top_y
        # Normalize the values
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        # Return the values
        return x_center, y_center, width, height

    def _gen_compose_images(self, foregrounds, background_image_path):
        # Open background image and convert to RGBA
        background_image = Image.open(background_image_path)
        background_image = background_image.convert('RGBA')

        # Resize background to desired width and height
        bg_width, bg_height = background_image.size

        if not self.fixed_image_sizes:
            # self.image_width = bg_width
            # self.image_height = bg_height
            # composite = background_image
            # use 60-100% of the background image
            rand_size = random.random() * 0.4 + 0.6
            self.image_width = int(bg_width * rand_size)
            self.image_height = int(bg_height * rand_size)
            composite = background_image.resize(
                (self.image_width, self.image_height), Image.LANCZOS)

        else:
            if bg_width >= self.image_width and bg_height >= self.image_height:
                crop_x_pos = random.randint(0, bg_width - self.image_width)
                crop_y_pos = random.randint(0, bg_height - self.image_height)
                composite = background_image.crop(
                    (crop_x_pos, crop_y_pos, crop_x_pos + self.image_width, crop_y_pos + self.image_height))
            else:

                composite = background_image.resize(
                    (self.image_width, self.image_height), PIL.Image.LANCZOS)

        annotations = dict()
        annotations['shapes'] = []
        annotations['imageWidth'] = self.image_width
        annotations['imageHeight'] = self.image_height

        fg_list = []

        fg_images = []
        mask_images = []
        foreground_positions = []
        fg_image_sizes = []
        fg_images_sized = []
        mask_images_sized = []

        for fg in foregrounds:
            fg_image = Image.open(fg['image_path'])

            # Resize foreground (based on https://github.com/basedrhys/cocosynth/commit/f0b5d4d97009a3a070ba9967ff536c7dd71af6ef)
            if not self.scale_foreground_by_background_size:
                # Apply random scale
                scale = random.random() * .5 + .5  # Pick something between .5 and 1
                new_size = (
                    int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
            else:
                # Scale the foreground based on the size of the resulting image
                min_bg_len = min(self.image_width, self.image_height)

                rand_length = random.random() * \
                    (self.scaling_factors[1] - self.scaling_factors[0]
                     ) + self.scaling_factors[0]
                long_side_len = rand_length * min_bg_len
                # Scale the longest side of the fg to be between the random length % of the bg
                if fg_image.size[0] > fg_image.size[1]:
                    long_side_dif = long_side_len / fg_image.size[0]
                    short_side_len = long_side_dif * fg_image.size[1]
                    new_size = (int(long_side_len), int(short_side_len))
                else:
                    long_side_dif = long_side_len / fg_image.size[1]
                    short_side_len = long_side_dif * fg_image.size[0]
                    new_size = (int(short_side_len), int(long_side_len))

            # This resizes the image and removes the alpha channel
            # fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)
            # with keeping the alpha channel
            fg_image = fg_image.resize(new_size, resample=Image.NEAREST)

            # Perform transformations
            fg_image = self._transform_foreground(fg_image)
            # TODO: random brightness and contrast
            # fg_image = self._random_brightness_and_contrast(fg_image)

            logger.debug(fg_image.mode)

            # Choose a random x,y position
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, f"""foreground {fg["image_path"]} is too big({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size({
                self.image_width}x{self.image_height}), check your input parameters"""
            foreground_position = (random.randint(
                0, max_x_position), random.randint(0, max_y_position))

            # avoid collisions of foreground objects (based on https://github.com/basedrhys/cocosynth/commit/d009a0de17b154ca3b469e8d4c0a7afa8fa51271)
            if self.avoid_collisions:
                fg_rect = [foreground_position[0],  # x1
                           foreground_position[1],  # y1
                           foreground_position[0] + fg_image.size[0],  # x2
                           foreground_position[1] + fg_image.size[1]]  # y2

                visited_centroids = []
                colliding_point = self._is_colliding(fg_rect, fg_list)

                while colliding_point is not None:
                    # Move the fg away from the colliding point
                    step_size = 50
                    curr_centroid_x = int((fg_rect[0] + fg_rect[2]) / 2)
                    curr_centroid_y = int((fg_rect[1] + fg_rect[3]) / 2)
                    new_centroid_pos = self._get_new_centroid_pos(colliding_point,
                                                                  (curr_centroid_x,
                                                                   curr_centroid_y),
                                                                  step_size)

                    if self._visited_point_before(new_centroid_pos, visited_centroids):
                        # print("Tried to re-visit point {}".format(new_centroid_pos))
                        logger.debug(
                            f'Tried to re-visit point {new_centroid_pos}')

                        fg_rect = None
                        break
                    visited_centroids.append(new_centroid_pos)

                    fg_rect = self._get_rect_position(
                        new_centroid_pos, fg_image)
                    colliding_point = self._is_colliding(fg_rect, fg_list)

                if fg_rect is None or self._outside_img(composite, fg_rect):
                    # print("Outside image {}".format(fg_rect))
                    continue
                else:
                    paste_position = (int(fg_rect[0]), int(fg_rect[1]))
                    fg_list.append(fg_rect)

            # random lighting conditions
            # fg_image = self._lighting_condition_transform(fg_image)

            # color harmonization
            if self.color_harmonization:
                if self.random_color_harmon_alpha:
                    fg_image = self._color_harmonization(
                        composite, fg_image, random.random())
                else:
                    fg_image = self._color_harmonization(
                        composite, fg_image, self.color_harmon_alpha)

            # Create a new foreground image as large as the composite and paste it on top
            fg_image_sized = Image.new(
                'RGBA', composite.size, color=(0, 0, 0, 0))
            fg_image_sized.paste(fg_image, foreground_position)

            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = fg_image.getchannel(3)
            alpha_mask_sized = Image.new('L', composite.size, color=0)
            alpha_mask_sized.paste(alpha_mask, foreground_position)

            # save the fg_image
            # fg_image.save(f'fg_{fg["category"]}.png')
            # alpha_mask.save(f'alpha_{fg["category"]}.png')

            # Grab the alpha pixels above a specified threshold
            alpha_threshold = 200
            mask_arr = np.array(np.greater(
                np.array(alpha_mask_sized), alpha_threshold), dtype=np.uint8)
            mask = np.float32(mask_arr)  # This is composed of 1s and 0s
            contours = measure.find_contours(
                mask, 0.5, positive_orientation='low')

            annotation = dict()
            annotation['points'] = []
            annotation['label'] = fg['category']
            annotation['group_id'] = None
            annotation['shape_type'] = "polygon"
            annotation['flags'] = {}

            for contour in contours:
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)

                if poly.area > 16:  # Ignore tiny polygons
                    if poly.geom_type == 'MultiPolygon':
                        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                        poly = poly.convex_hull

                    # Ignore if still not a Polygon (could be a line or point)
                    if poly.geom_type == 'Polygon':
                        segmentation = list(
                            zip(*reversed(poly.exterior.coords.xy)))
                        annotation['points'].extend(segmentation)
                else:
                    warnings.warn(
                        f'Polygon area is too small: {poly.area}. Ignoring...')
            annotations['shapes'].append(annotation)

            fg_images.append(fg_image)
            mask_images.append(alpha_mask)
            foreground_positions.append(foreground_position)
            fg_image_sizes.append(fg_image.size)
            fg_images_sized.append(fg_image_sized)
            mask_images_sized.append(alpha_mask_sized)

        return fg_images, mask_images, composite, annotations, foreground_positions, fg_image_sizes, fg_images_sized, mask_images_sized

    def _paste_foregrounds(self, fg_images, mask_images, composite, foreground_positions, fg_image_sizes, fg_images_sized, mask_images_sized):
        # Composite the foreground image onto the background
        blending_mode_images = []
        from copy import deepcopy
        # for each blending mode
        for blending_mode in self.blending_methods:
            img = deepcopy(composite)
            for fg_image, mask_image, foreground_position, fg_image_size, fg_image_sized, mask_image_sized in zip(fg_images, mask_images, foreground_positions, fg_image_sizes, fg_images_sized, mask_images_sized):
                if blending_mode == BlendingMode.ALPHA_BLENDING:
                    img = Image.composite(
                        fg_image_sized, img, mask_image_sized)
                elif blending_mode == BlendingMode.GAUSSIAN_BLUR:
                    # Convert new_alpha_mask to a NumPy array
                    mask_arr = np.array(mask_image_sized)

                    # Apply Gaussian blur to the mask
                    blurred_mask = cv2.GaussianBlur(
                        mask_arr, (self.gaussian_options[0], self.gaussian_options[0]), sigmaX=self.gaussian_options[1], sigmaY=self.gaussian_options[1])

                    img = Image.composite(
                        fg_image_sized, img, Image.fromarray(blurred_mask))
                elif blending_mode == BlendingMode.POISSON_BLENDING_NORMAL or blending_mode == BlendingMode.POISSON_BLENDING_MIXED:
                    center = (
                        foreground_position[0] + fg_image_size[0] // 2,
                        foreground_position[1] + fg_image_size[1] // 2
                    )

                    img = poisson_blend_rgba(
                        fg_image, img, mask_image, center, blending_mode)

                elif blending_mode == BlendingMode.PYRAMID_BLEND:
                    img = pyramid_blend(
                        fg_image_sized, img, mask_image_sized, self.pyramid_blending_levels)

            # cv2.imshow(blending_mode.value, np.array(img))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            blending_mode_images.append({
                'image': img,
                'blending_mode': blending_mode.value
            })

        return blending_mode_images

    def _color_harmonization(self, target, source, alpha):
        """
        Perform color transformation with an adjustable blending factor.

        Parameters:
        - source: Source image as a NumPy array.
        - target: Target image as a NumPy array.

        Returns:
        - Transformed source image as a PIL Image in RGBA mode.
        """
        source = np.array(source)
        source_a = source[:, :, 3]  # Extract alpha channel to paste back later
        target = np.array(target)

        # Convert to Lab color space
        source = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_RGBA2RGB)
        source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(
            np.float32) / 255.0
        target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(
            np.float32) / 255.0

        # Split channels
        s_l, s_a, s_b = source[:, :, 0], source[:, :, 1], source[:, :, 2]
        t_l, t_a, t_b = target[:, :, 0], target[:, :, 1], target[:, :, 2]

        # Normalize source channels
        s_l = (s_l - np.mean(s_l)) / np.std(s_l)
        s_a = (s_a - np.mean(s_a)) / np.std(s_a)
        s_b = (s_b - np.mean(s_b)) / np.std(s_b)

        # Scale by target statistics
        s_l = s_l * np.std(t_l) + np.mean(t_l)
        s_a = s_a * np.std(t_a) + np.mean(t_a)
        s_b = s_b * np.std(t_b) + np.mean(t_b)

        # Blend between original and transformed values
        s_l = alpha * s_l + (1 - alpha) * source[:, :, 0]
        s_a = alpha * s_a + (1 - alpha) * source[:, :, 1]
        s_b = alpha * s_b + (1 - alpha) * source[:, :, 2]

        # Merge channels back
        source = np.dstack([s_l, s_a, s_b]) * 255.0
        source = np.clip(source, 0, 255).astype(np.uint8)

        # Convert back to RGBA
        source = cv2.cvtColor(source, cv2.COLOR_LAB2RGB)

        source = cv2.cvtColor(source, cv2.COLOR_RGB2RGBA)
        source[:, :, 3] = source_a  # Restore alpha channel

        return Image.fromarray(source, mode='RGBA')

    @staticmethod
    def _get_point_to_move_from(colliding_centroids):
        """
        Average all the centroid locations that the fg was colliding with
        This gives a point for the fg to move away from
        input: array of centroids -> [(200, 100), (50, 20), ...]
        """
        return np.mean(colliding_centroids, 0)

    def _is_colliding(self, fg_rect, fg_rect_list):
        """
        Check if the current foreground object is colliding with any foregrounds
        we've placed on the image already. The overlap threshold controls how much overlap
        of the rectangles is allowed. The overlap is a proportion of the total area of this
        foreground object, as opposed to an absolute value
        """
        overlap_thresh = 0.3

        colliding_centroids = []

        for rect in fg_rect_list:
            x1, y1, x2, y2 = fg_rect
            comp_x1, comp_y1, comp_x2, comp_y2 = rect
            x_overlap = max(0, min(comp_x2, x2) - max(comp_x1, x1))
            y_overlap = max(0, min(comp_y2, y2) - max(comp_y1, y1))

            rect_area = (x2 - x1) * (y2 - y1)
            total_overlap_area = x_overlap * y_overlap

            prop_overlap = total_overlap_area / float(rect_area)

            if prop_overlap > overlap_thresh:
                colliding_centroid = (comp_x2 - comp_x1, comp_y2 - comp_y1)
                colliding_centroids.append(colliding_centroid)

        if len(colliding_centroids) == 0:
            return None
        else:
            return self._get_point_to_move_from(colliding_centroids)

    def _random_brightness_and_contrast(self, fg_image):
        """
        Randomly change the brightness of the image
        """
        # only for visible pixels
        alpha = fg_image.getchannel(3)
        fg_image = fg_image.convert('RGB')
        fg_image = A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5)(image=np.array(fg_image))['image']
        # random contrast
        fg_image = A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5)(image=np.array(fg_image))['image']

        # convert back to RGBA
        fg_image = Image.fromarray(fg_image)
        # add alpha channel back
        fg_image.putalpha(alpha)
        return fg_image

    @staticmethod
    def _get_new_centroid_pos(pt_a, pt_b, step_size):
        """
        https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
        Given the point to move from (pt_a), and our current centroid (pt_b),
        move further away along the line i.e.:
        a -------- b --<step_size>-- new_point
        """
        v = np.subtract(pt_b, pt_a)
        norm_v = v / np.linalg.norm(v)

        # Convert to an int tuple
        new_point = tuple(map(int, pt_b + (step_size * norm_v)))
        return new_point

    @staticmethod
    def _get_rect_position(centroid, fg_image):
        """
        Convert a centroid to the corresponding rectangle, given a foreground image
        """
        width, height = fg_image.size
        x1 = centroid[0] - (width / 2)
        y1 = centroid[1] - (height / 2)
        x2 = centroid[0] + (width / 2)
        y2 = centroid[1] + (height / 2)
        return [x1, y1, x2, y2]

    @staticmethod
    def _visited_point_before(coll_pt, coll_pts):
        """
        Check if we've visited this point before.
        Stops collision avoidance from getting stuck in between a group of points
        """
        for pt in coll_pts:
            if coll_pt[0] == pt[0] and coll_pt[1] == pt[1]:
                return True

        return False

    @staticmethod
    def _outside_img(img, fg_rect):
        """
        Don't paste the foreground if the centroid is outside the image
        """
        curr_centroid_x = int((fg_rect[0] + fg_rect[2]) / 2)
        curr_centroid_y = int((fg_rect[1] + fg_rect[3]) / 2)

        return (curr_centroid_x < 0 or
                curr_centroid_x > img.size[0] or
                curr_centroid_y < 0 or
                curr_centroid_y > img.size[1])

    def _transform_foreground(self, fg_image):
        if self.transforms:
            return Image.fromarray(self.transforms(image=np.array(fg_image))['image'])
        return fg_image

    def _make_output_dir(self):
        # check if the output directory contains any files
        if len(list(self.output_dir.glob('*'))) > 0 and not self.overwrite_output:
            raise Exception(
                f'Output directory {self.output_dir} is not empty. Use --overwrite_output to overwrite it.')
        # clear the output directory
        if self.overwrite_output and self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if self.output_mode == OutputMode.CLASSIFICATION_SINGLE:
            # Create directories for images and labels
            self.output_dir.mkdir(exist_ok=True)
            # Create directories for classes
            for category in self.categories:
                (self.output_dir / category['name']).mkdir(exist_ok=True)
        elif self.output_mode == OutputMode.YOLO:
            (self.output_dir / 'images').mkdir(exist_ok=True)
            (self.output_dir / 'labels').mkdir(exist_ok=True)
        elif self.output_mode == OutputMode.COCO:
            self.output_dir.mkdir(exist_ok=True)

    def generate_images(self):
        # Create directories for images and labels
        # self._make_output_dir()
        # quit()
        if self.parallelize:
            Parallel(n_jobs=4)(delayed(self._generate_image)(
                i) for i in tqdm(range(1, self.image_number + 1)))
        else:
            for i in tqdm(range(1, self.image_number + 1)):
                self._generate_image(i)
