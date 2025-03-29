import cv2
import numpy as np
from PIL import Image
from SynDataGenYOLO.utils.modes import BlendingMode


def poisson_blend_rgba(fg_image, bg_image, mask_image, center, blending_mode):
    """
    Perform Poisson blending for RGBA images.

    Args:
        fg_image (PIL.Image): RGBA Foreground image.
        bg_image (PIL.Image): RGBA Background image.
        mask_image (PIL.Image): Binary mask image where the foreground is white (255) and the rest is black (0).
        position (tuple): (x, y) coordinates where the foreground is placed on the background.

    Returns:
        PIL.Image: Blended RGBA image.
    """
    # Separate alpha channels
    fg_array = np.array(fg_image)
    bg_array = np.array(bg_image)
    mask_array = np.array(mask_image)

    # Extract RGB channels
    fg_rgb = fg_array[:, :, :3]
    bg_rgb = bg_array[:, :, :3]
    # alpha_fg = fg_array[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

    # Ensure mask is binary
    mask_array = np.where(mask_array > 128, 255, 0).astype(np.uint8)

    # Convert images to OpenCV format (BGR)
    fg_bgr = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR)
    bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

    # Perform Poisson blending on the RGB channels
    blended_bgr = cv2.seamlessClone(fg_bgr, bg_bgr, mask_array, center, cv2.NORMAL_CLONE if blending_mode ==
                                    BlendingMode.POISSON_BLENDING_NORMAL else cv2.MIXED_CLONE)
    # Convert back to RGB
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    # Convert back to PIL image
    blended_image = Image.fromarray(blended_rgb, mode="RGB")

    return blended_image


def pyramid_blend(source, target, mask, num_levels=5):
    # 1. as in api55's answer, mask needs to be from 0 to 1, since you're multiplying a pixel value by it. Since mask
    # is binary, we only need to set all values which are 255 to 1
    # make mask from 0 to 1
    # image object is a PIL image
    mask = np.array(mask)  # (480, 640)
    # convert the mask to a rgb image
    mask = np.stack((mask, mask, mask), axis=2)  # (480, 640, 3)
    source = np.array(source)
    target = np.array(target)
    source = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
    target = cv2.cvtColor(target, cv2.COLOR_RGBA2RGB)

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0

    # Initialize Gaussian pyramids for the two images and the mask
    GA = source.copy()
    GB = target.copy()
    GM = mask.copy()

    # Generate the Gaussian pyramids
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]

    for i in range(num_levels):
        # Downsample
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)

        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # Initialize Laplacian pyramids
    # Start with the smallest Gaussian level
    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]

    # Build Laplacian pyramids by subtracting successive Gaussian levels
    for i in range(num_levels - 1, 0, -1):
        # Get the size of the next level
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])

        # Compute the Laplacian by subtracting the upsampled Gaussian level from the current level
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i], dstsize=size))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i], dstsize=size))

        # Append Laplacians to their respective pyramids
        lpA.append(LA)
        lpB.append(LB)

        # Append the corresponding Gaussian mask level
        gpMr.append(gpM[i - 1])

    # Blend the Laplacian pyramids using the Gaussian mask
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        # Perform weighted blending for each level
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # Reconstruct the final blended image by collapsing the pyramid
    ls_ = LS[0]
    for i in range(1, num_levels):
        # Get the size of the current level
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size),
                      np.float32(LS[i]))  # Add upsampled levels

        # Clip values to the range [0, 255] to avoid overflow when converting to uint8
        ls_ = cv2.normalize(ls_, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the final blended image to uint8 for display or saving
    return Image.fromarray(np.uint8(ls_), 'RGB')
