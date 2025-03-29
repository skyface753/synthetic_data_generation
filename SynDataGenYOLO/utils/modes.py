from enum import Enum

# enum with the blending modes, including the strings


class BlendingMode(Enum):
    ALPHA_BLENDING = 'ALPHA'
    POISSON_BLENDING_NORMAL = 'POISSON_NORMAL'
    POISSON_BLENDING_MIXED = 'POISSON_MIXED'
    GAUSSIAN_BLUR = 'GAUSSIAN'
    PYRAMID_BLEND = 'PYRAMID'


class OutputMode(Enum):
    YOLO = 'YOLO'
    COCO = 'COCO'
    CLASSIFICATION_SINGLE = 'CLASSIFICATION_SINGLE'
