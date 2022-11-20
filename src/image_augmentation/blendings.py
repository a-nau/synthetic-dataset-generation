import random
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

root_dir = Path(__file__).parent.parent.parent

from src.generator.utils import PIL2array1C, PIL2array3C
from src.image_augmentation.pb import create_mask, poisson_blend
from src.image_augmentation.gamma_correction import adjust_gamma_of_image


def apply_blendings_and_paste_onto_background(
    backgrounds, blending_list, foreground, mask, x, y
):
    for i in range(len(blending_list)):
        new_foreground = foreground.copy()
        new_mask = mask.copy()
        # Cases
        if blending_list[i] in ["none", "motion"]:
            pass  # both are only copied
        elif blending_list[i] == "gaussian":
            new_mask = Image.fromarray(
                cv2.GaussianBlur(PIL2array1C(new_mask), (5, 5), 2)
            )
        elif blending_list[i] == "box":
            new_mask = Image.fromarray(cv2.blur(PIL2array1C(new_mask), (3, 3)))
        elif blending_list[i].startswith("poisson"):
            if blending_list[i] == "poisson":
                backgrounds[i] = apply_poisson_blending(
                    new_foreground, new_mask, backgrounds[i], (y, x)
                )
            else:
                try:
                    backgrounds[i] = apply_poisson_blending_fast(
                        new_foreground, new_mask, backgrounds[i], (y, x)
                    )
                except Exception as e:
                    print(f"Error: {e}; are you sure you have CUDA enabled?")
            continue
        elif blending_list[i] == "gamma_correction":
            new_foreground = apply_gamma_correction(new_foreground)
        elif blending_list[i] == "illumination":
            new_foreground = apply_illumination_change(new_foreground, new_mask)
        elif blending_list[i] == "mixed":
            new_foreground = apply_gamma_correction(new_foreground)
            new_foreground = apply_illumination_change(new_foreground, new_mask)
            new_mask = apply_random_mask_adjustment(new_mask)
        else:
            raise NotImplementedError(
                f"Could not find blending of type: {blending_list[i]}"
            )
        backgrounds[i].paste(new_foreground, (x, y), new_mask)


def apply_poisson_blending(foreground, mask, background, offset):
    (
        img_mask,
        img_src,
        img_target,
        offset_adj,
    ) = create_temporary_input_for_poisson_blending(
        background, foreground, mask, offset
    )
    blend_method = "normal"  # random.choice(['normal', 'mixed'])
    background_array = poisson_blend(
        img_mask, img_src, img_target, method=blend_method, offset_adj=offset_adj
    )
    new_background = Image.fromarray(background_array, "RGB")
    return new_background


def create_temporary_input_for_poisson_blending(background, foreground, mask, offset):
    img_mask = PIL2array1C(mask)
    img_src = PIL2array3C(foreground).astype(np.float64)
    img_target = PIL2array3C(background)
    img_mask, img_src, offset_adj = create_mask(
        img_mask.astype(np.float64), img_target, img_src, offset=offset
    )
    return img_mask, img_src, img_target, offset_adj


def apply_poisson_blending_fast(foreground, mask, background, offset, backend="cuda"):
    (
        img_mask,
        img_src,
        img_target,
        offset_adj,
    ) = create_temporary_input_for_poisson_blending(
        background, foreground, mask, offset
    )
    tmp_dir = root_dir / "tmp" / str(uuid.uuid4())
    tmp_dir.parent.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    mask_path = tmp_dir / "mask.jpg"
    src_path = tmp_dir / "src.jpg"
    target_path = tmp_dir / "target.jpg"
    result_path = tmp_dir / "result.jpg"
    cv2.imwrite(mask_path.as_posix(), img_mask * 255)
    cv2.imwrite(
        src_path.as_posix(), cv2.cvtColor(img_src.astype(np.uint8), cv2.COLOR_BGR2RGB)
    )
    cv2.imwrite(
        target_path.as_posix(),
        cv2.cvtColor(img_target.astype(np.uint8), cv2.COLOR_BGR2RGB),
    )
    cmd = f"fpie -s {src_path} -m {mask_path} -t {target_path} -o {result_path.resolve()} -h1 {offset_adj[0]} -w1 {offset[1]} -b {backend} -n 5000 -g src"
    process = subprocess.Popen(cmd.split(" "))
    process.wait()
    new_background = Image.open(result_path)
    shutil.rmtree(tmp_dir)
    return new_background


def apply_illumination_change(img, mask):
    alpha = 1.75 + ((random.random() - 0.25) * 1)
    beta = (random.random()) * 0.3
    foreground = cv2.illuminationChange(
        PIL2array3C(img), PIL2array1C(mask), alpha=alpha, beta=beta
    )
    foreground = Image.fromarray(foreground, "RGB")
    return foreground


def apply_gamma_correction(img):
    img = adjust_gamma_of_image(PIL2array3C(img), 1 + ((random.random() + 0.5) * 0.25))
    img = Image.fromarray(img, "RGB")
    return img


def apply_random_mask_adjustment(mask):
    choice = random.choice(["none", "gaussian", "blur"])
    if choice == "gaussian":
        mask = Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask), (3, 3), 2))
    elif choice == "box":
        mask = Image.fromarray(cv2.blur(PIL2array1C(mask), (3, 3)))
    else:
        pass
    return mask
