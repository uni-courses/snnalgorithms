"""Created with chatGTP, zooms in a region of an image."""

import os
from typing import Tuple

from PIL import Image
from typeguard import typechecked


@typechecked
def copy_region_of_img(
    *,
    src_path: str,
    dst_dir: str,
    x_coords: Tuple[float, float],
    y_coords: Tuple[float, float],
) -> None:
    """Copy a region from a .png file and save the zoomed image in the
    destination directory.

    Parameters:
    - src_path: str: The path to the source .png file.
    - dst_dir: str: The destination directory.
    - x_left: int: fraction [0-1] for left side of the region to copy.
    - x_right: int: fraction [0-1] for right side of the region to copy.
    - y_low: int: fraction [0-1] for bottom side of the region to copy.
    - y_top: int: fraction [0-1] for the top right side of the region to copy.
    """
    # Check if the source file is a .png file
    if not src_path.endswith(".png"):
        raise ValueError("The source file must be a .png file.")

    x_left, x_right = x_coords
    y_low, y_top = y_coords

    # Open the image file
    with Image.open(src_path) as img:

        width, height = img.size

        # Copy the specified region from the source image
        region = img.crop(
            (
                x_left * width,
                height * (1 - y_top),
                x_right * width,
                height * (1 - y_low),
            )
        )

        # Get the base name of the source file.
        src_name = os.path.splitext(os.path.basename(src_path))[0]

        # Create the destination path for the zoomed image
        dst_path = os.path.join(dst_dir, "zoomed_" + src_name + ".png")

        # Save the zoomed image to the destination file
        region.save(dst_path)
