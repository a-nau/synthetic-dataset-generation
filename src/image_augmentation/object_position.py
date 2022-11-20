import random

from src.config import MAX_TRUNCATION_FRACTION, MAX_ATTEMPTS_TO_SYNTHESIZE
from src.models.auxiliary import Rectangle
from src.image_augmentation.misc import overlap


def find_valid_object_position(
    already_syn, dontocclude, h, o_h, o_w, w, xmax, xmin, ymax, ymin
):
    attempt = 0
    while True:
        attempt += 1
        x = random.randint(
            int(-MAX_TRUNCATION_FRACTION * o_w),
            int(w - o_w + MAX_TRUNCATION_FRACTION * o_w),
        )
        y = random.randint(
            int(-MAX_TRUNCATION_FRACTION * o_h),
            int(h - o_h + MAX_TRUNCATION_FRACTION * o_h),
        )
        if dontocclude:
            found = True
            for prev in already_syn:
                ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                rb = Rectangle(x + xmin, y + ymin, x + xmax, y + ymax)
                if overlap(ra, rb):
                    found = False
                    break
            if found:
                break
        else:
            break
        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
            break
    if dontocclude:
        already_syn.append([x + xmin, x + xmax, y + ymin, y + ymax])
    return x, y, attempt
