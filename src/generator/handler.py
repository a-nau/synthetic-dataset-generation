import json
import random
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Dict, List

import tqdm

from src.config import (
    OBJECT_CATEGORIES,
    BLENDING_LIST,
    NUMBER_OF_WORKERS,
    MIN_NO_OF_OBJECTS,
    MAX_NO_OF_OBJECTS,
    MIN_NO_OF_DISTRACTOR_OBJECTS,
    MAX_NO_OF_DISTRACTOR_OBJECTS,
)
from src.generator.create import create_image_anno_wrapper
from src.generator.join_annotations import (
    save_joined_mscoco_annotation_file_from_paths_of_single_image_annotations,
)
from src.generator.utils import init_worker
from src.models.img_data import ImgDataRGBA, BaseImgData


def generate_synthetic_dataset(
    output_dir: str,
    object_json: str,
    distractor_json: str,
    background_json: str,
    number_of_images: Dict,
    dontocclude: bool,
    rotation: bool,
    scale: bool,
    multithreading: bool,
):
    """
    Generate synthetic dataset
    :param output_dir: output directory path
    :param object_json: path to objects of interest json
    :param distractor_json: path to distractor object json
    :param background_json: path to background json
    :param number_of_images: for each split contains the number of images
    :param dontocclude: disable occlusion
    :param rotation: enable rotation of objects
    :param scale: enable scaling of objects
    :param multithreading: use multithreading
    """

    for split_type in ["test", "train", "validation"]:
        print(f"{'#' * 20} Generating {split_type} data {'#' * 20}")
        start_time = time.time()
        (
            background_files,
            distractor_data,
            objects_data,
            labels,
            split_output_dir,
        ) = load_relevant_data(
            output_dir, object_json, distractor_json, background_json, split_type,
        )
        full_anno_list, full_img_list, params_list = create_list_of_img_configurations(
            objects_data,
            distractor_data,
            background_files,
            OBJECT_CATEGORIES,
            split_output_dir,
            number_of_images[split_type],
        )

        render_configurations(
            full_anno_list,
            params_list,
            split_output_dir,
            dontocclude,
            rotation,
            scale,
            multithreading,
        )
        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print(f"Generation of {split_type}: {elapsed:.2f} min")


def load_relevant_data(
    output_dir: str,
    object_json: str,
    distractor_json: str,
    background_json: str,
    split_type: str,
):
    output_dir = (Path(output_dir) / split_type).resolve()
    output_dir.mkdir(exist_ok=True)
    # Objects
    object_files = load_data_from_split_file(object_json, split_type)
    labels = ["box"] * len(object_files)
    objects_data = [ImgDataRGBA(object_files[i], labels[i]) for i in range(len(labels))]
    random.shuffle(objects_data)
    # Distractors
    distractor_files = load_data_from_split_file(distractor_json, split_type)
    distractor_data = [
        ImgDataRGBA(distractor_files[i], OBJECT_CATEGORIES[1]["name"])
        for i in range(len(distractor_files))
    ]
    random.shuffle(distractor_data)
    # Backgrounds
    background_files = load_data_from_split_file(background_json, split_type)
    random.shuffle(background_files)
    return background_files, distractor_data, objects_data, labels, output_dir


def load_data_from_split_file(json_file: Union[str, Path], split_type: str):
    if isinstance(json_file, str):
        json_file = Path(json_file)
    assert json_file.exists(), f"File {json_file.resolve()} does not exist!"
    with json_file.open("r") as f:
        data = json.load(f)
    if data.get("path", None) is not None:
        base_path = Path(data["path"])
        base_path = base_path if base_path.exists() else json_file.parent
        return [base_path / f for f in data[split_type]]
    else:
        return [json_file.parent / f for f in data[split_type]]


def render_configurations(
    full_anno_list,
    params_list,
    output_dir: Path,
    dontocclude: bool,
    rotation_augment: bool,
    scale_augment: bool,
    multithreading: bool,
):
    # Run configurations
    partial_func = partial(
        create_image_anno_wrapper,
        scale_augment=scale_augment,
        rotation_augment=rotation_augment,
        blending_list=BLENDING_LIST,
        dontocclude=dontocclude,
    )
    print(f"Found {len(params_list)} params lists")

    if not multithreading:
        for p in tqdm.tqdm(params_list):
            partial_func(p)
    else:
        p = Pool(NUMBER_OF_WORKERS, init_worker)
        try:
            p.map(partial_func, params_list)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    save_joined_mscoco_annotation_file_from_paths_of_single_image_annotations(
        sum(full_anno_list, []), output_dir.parent / f"{output_dir.name}.json",
    )


def create_list_of_img_configurations(
    objects_data: List[BaseImgData],
    distractors_data: List[BaseImgData],
    background_files: List[str],
    categories,
    output_dir: Path,
    num_images: int,
):
    idx = 0
    params_list = []
    full_img_list = []
    full_anno_list = []
    for _ in range(num_images):
        objects = []
        distractor_objects = []

        # Get list of objects
        n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(objects_data))
        for i in range(n):
            objects.append(random.choice(objects_data))
        # Get list of distractor objects
        if len(distractors_data) > 0:
            n = min(
                random.randint(
                    MIN_NO_OF_DISTRACTOR_OBJECTS, MAX_NO_OF_DISTRACTOR_OBJECTS
                ),
                len(distractors_data),
            )
            for i in range(n):
                distractor_objects.append(random.choice(distractors_data))

        idx += 1
        bg_file = random.choice(background_files)
        img_files = []
        anno_files = []
        img_dir = output_dir / str(idx).zfill(5)
        img_dir.mkdir(exist_ok=True)
        for blending_type in BLENDING_LIST:
            i = 0
            img_file = img_dir / f"image_{blending_type}{str(i).zfill(2)}.jpg"
            anno_file = img_file.with_suffix(".json")
            while img_file in img_files:
                i += 1
                img_file = img_dir / f"image_{blending_type}{str(i).zfill(2)}.jpg"
                anno_file = img_file.with_suffix(".json")
            img_files.append(img_file)
            anno_files.append(anno_file)
        params = {
            "objects": objects,
            "distractor_objects": distractor_objects,
            "img_files": img_files,
            "anno_files": anno_files,
            "bg_file": bg_file,
            "categories": categories,
        }
        params_list.append(params)
        full_img_list.append(img_files)
        full_anno_list.append(anno_files)
    return full_anno_list, full_img_list, params_list
