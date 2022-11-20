import random

from PIL import Image

from src.config import MIN_SCALE, MAX_SCALE, MAX_UPSCALING


def augment_rotation(foreground, h, mask, max_degrees, w):
    while True:
        rot_degrees = random.randint(-max_degrees, max_degrees)
        foreground_tmp = foreground.rotate(rot_degrees, expand=True)
        mask_tmp = mask.rotate(rot_degrees, expand=True)
        o_w, o_h = foreground_tmp.size
        if w - o_w > 0 and h - o_h > 0:
            break
    mask = mask_tmp
    foreground = foreground_tmp
    return foreground, mask, o_h, o_w


def augment_scale(foreground, bg_h, mask, fg_h, fg_w, bg_w):
    width_scale = fg_w / bg_w
    height_scale = fg_h / bg_h
    choosen_scale = max(
        width_scale, height_scale
    )  # scale between foreground and background
    while True:
        scale = random.uniform(MIN_SCALE, MAX_SCALE) * (1 / choosen_scale)
        scale = min(
            scale, MAX_UPSCALING
        )  # allow only certain upscaling to prevent blurry foregrounds
        o_w, o_h = int(scale * fg_w), int(scale * fg_h)
        if bg_w - o_w > 0 and bg_h - o_h > 0 and o_w > 0 and o_h > 0:
            break
    foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
    mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
    return foreground, mask, o_h, o_w
