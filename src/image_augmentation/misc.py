import numpy as np

from src.config import MAX_ALLOWED_IOU
from src.generator.utils import PIL2array1C
from src.models.auxiliary import ImgSize, ImgPosition


def create_full_size_and_sharpened_mask(
    mask, original_size: ImgSize, paste_position: ImgPosition, threshold=200
):
    mask = PIL2array1C(mask)
    full_mask = np.zeros((original_size.height, original_size.width))
    # sharpen mask
    sharp_mask = np.zeros_like(mask)
    sharp_mask[mask > threshold] = 255
    sharp_mask[mask <= threshold] = 0
    # Find start and end positions in original image
    start_y = max(0, paste_position.y)
    end_y = min(original_size.height, paste_position.y + mask.shape[0])
    start_x = max(0, paste_position.x)
    end_x = min(original_size.width, paste_position.x + mask.shape[1])
    # Find start and end positions in mask
    start_mask_y = max(0, -paste_position.y)
    start_mask_x = max(0, -paste_position.x)
    end_mask_y = min(mask.shape[0], start_mask_y + (end_y - start_y))
    end_mask_x = min(mask.shape[1], start_mask_x + (end_x - start_x))
    # Paste mask onto image
    full_mask[start_y:end_y, start_x:end_x] = sharp_mask[
        start_mask_y:end_mask_y, start_mask_x:end_mask_x
    ]
    return full_mask.astype(np.uint8)


def adjust_masks_for_occlusion(masks):
    foreground_mask = np.zeros_like(masks[0]).astype(np.uint8)
    for i in reversed(list(range(len(masks)))):
        # remove occluded part from mask
        occluded_mask = masks[i] - foreground_mask
        occluded_mask[occluded_mask < 0] = 0
        occluded_mask = occluded_mask.astype(np.uint8)
        masks[i] = occluded_mask
        # update foreground masks
        foreground_mask[occluded_mask == 255] = 255


def overlap(a, b):
    """Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    """
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (
        (dx >= 0)
        and (dy >= 0)
        and float(dx * dy) > MAX_ALLOWED_IOU * (a.xmax - a.xmin) * (a.ymax - a.ymin)
    ):
        return True
    else:
        return False
