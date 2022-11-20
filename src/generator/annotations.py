import json
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt  # noqa
import numpy as np
from PIL import Image

from src.config import IGNORE_LABELS


def get_bbox_and_segmentation_of_single_object(mask, image_size):
    mask_size = mask.shape[::-1]  # image_size: (w, h)
    mask[mask < 250] = 0  # somehow there are values with 1 here
    assert mask_size == image_size
    contour_res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Depending on OpenCV version output has 2 or 3 components
    if len(contour_res) == 2:
        contours = contour_res[0]
    elif len(contour_res) == 3:
        contours = contour_res[1]
    else:
        raise NotImplementedError(
            "Unknown OpenCV version! Result of findContours unexpected."
        )

    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        return None

    pos = np.where(mask)
    pos_min = np.min(pos, axis=1)[::-1]
    pos_max = np.max(pos, axis=1)[::-1]
    size = pos_max - pos_min
    bbox = pos_min.tolist() + size.tolist()

    return segmentation, bbox, 0  # todo: calc area


def create_image_and_annotation_dict_mscoco(
    image_path: Path,
    image_id_int: int,
    masks: List[np.ndarray],
    mask_category_ids: List[int],
):
    image = Image.open(image_path)
    width, height = image.size
    image_dict = {
        "id": image_id_int,
        "file_name": "/".join(image_path.parts[-3:]),
        "width": width,
        "height": height,
    }

    # compute mask and bounding box
    annotation_dicts = []
    for mask_id_int, mask in enumerate(masks):
        mask_info = get_bbox_and_segmentation_of_single_object(
            mask.copy(), (width, height)
        )
        if not mask_info:
            print(f"Could not find mask with ID: {mask_id_int}!")
            continue
        segmentations, bbox, area = mask_info
        annotation_dict = {
            "segmentation": segmentations,
            "iscrowd": 0,
            "image_id": image_id_int,
            "category_id": mask_category_ids[mask_id_int],  # all have category box
            "id": mask_id_int,
            "bbox": bbox,
            "area": area,
        }
        annotation_dicts.append(annotation_dict)
    return image_dict, annotation_dicts


def save_single_annotation_data_to_json(
    img_dict, annotation_dicts, categories, render_config, output_file_path
):
    subset_dict = {
        "categories": categories,
        "annotations": annotation_dicts,
        "images": [img_dict],
        "render_config": {
            "path": {
                object_type: (
                    str(render_config[object_type][0].img_path.parent)
                    if len(render_config[object_type]) > 0
                    else ""
                )
                for object_type in ["objects", "distractor_objects"]
            },
            "objects": [f.img_path.name for f in render_config["objects"]],
            "distractors": [
                f.img_path.name for f in render_config["distractor_objects"]
            ],
            "background": str(render_config["bg_file"]),
        },
    }
    with open(output_file_path, "w") as f:
        json.dump(subset_dict, f)


def save_annotation_data_to_json(
    imgs_dict, annotation_dicts, categories, output_file_path
):
    # check images_list
    mask_image_id_set = set(
        [annotation_dict["image_id"] for annotation_dict in annotation_dicts]
    )
    images_list = [
        image for image in imgs_dict.values() if image["id"] in mask_image_id_set
    ]

    # categories
    subset_dict = {
        "categories": categories,
        "annotations": annotation_dicts,
        "images": images_list,
    }
    with open(output_file_path, "w") as f:
        json.dump(subset_dict, f)
    pass


def remove_ignore_label_segmentations(annotation_dicts):
    del_ids = []
    for i, anno_dict in enumerate(annotation_dicts):
        if anno_dict["category_id"] in IGNORE_LABELS:
            del_ids.append(i)
    for i in reversed(del_ids):
        del annotation_dicts[i]
