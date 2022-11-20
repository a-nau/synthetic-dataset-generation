from PIL import Image

from src.config import MAX_DEGREES, MAX_ATTEMPTS_TO_SYNTHESIZE
from src.generator.annotations import (
    create_image_and_annotation_dict_mscoco,
    save_single_annotation_data_to_json,
    remove_ignore_label_segmentations,
)
from src.generator.utils import PIL2array3C
from src.image_augmentation.basic_augmentations import (
    augment_scale,
    augment_rotation,
)
from src.image_augmentation.blendings import apply_blendings_and_paste_onto_background
from src.image_augmentation.misc import (
    create_full_size_and_sharpened_mask,
    adjust_masks_for_occlusion,
)
from src.image_augmentation.motion_blur import LinearMotionBlur3C
from src.image_augmentation.object_position import find_valid_object_position
from src.models.auxiliary import ImgSize, ImgPosition


def create_image_anno_wrapper(
    args,
    scale_augment=False,
    rotation_augment=False,
    blending_list=["none"],
    dontocclude=False,
):
    """ Wrapper used to pass params to workers
    """
    categories = args["categories"]
    del args["categories"]
    anno_files = args["anno_files"]
    del args["anno_files"]
    # Create synthesized images, including masks and labels
    img_files, masks, mask_category_ids = create_image_anno(
        scale_augment=scale_augment,
        rotation_augment=rotation_augment,
        blending_list=blending_list,
        dontocclude=dontocclude,
        **args
    )
    # Generate MS COCO style annotations from these and save
    for i in range(len(img_files)):
        img_dict, annotation_dicts = create_image_and_annotation_dict_mscoco(
            image_path=args["img_files"][i],
            image_id_int=i,
            masks=masks,
            mask_category_ids=mask_category_ids,
        )
        remove_ignore_label_segmentations(annotation_dicts)
        save_single_annotation_data_to_json(
            img_dict, annotation_dicts, categories, args, anno_files[i]
        )
    return


def create_image_anno(
    objects,
    distractor_objects,
    img_files,
    bg_file,
    scale_augment=False,
    rotation_augment=False,
    blending_list=["none"],
    dontocclude=False,
):
    """Add data augmentation, synthesizes images and generates annotations according to given parameters

    Args:
        objects(list): List of objects whose annotations are also important
        distractor_objects(list): List of distractor objects that will be synthesized but whose annotations are not required
        img_files(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path
        bg_w(int): Width of synthesized image
        bg_h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dontocclude(bool): Generate images with occlusion
    """

    all_objects = objects + distractor_objects
    already_syn = []
    assert len(all_objects) > 0
    while True:  # creating new attempts for synthesizing
        masks = []
        mask_category_ids = []
        backgrounds = []

        # Load background (can be RGB or RGBA)
        background_rgba = Image.open(bg_file).convert("RGBA")
        background = Image.new("RGBA", background_rgba.size, (255, 255, 255))
        background = Image.alpha_composite(background, background_rgba).convert("RGB")

        bg_w, bg_h = background.size
        # background = background.resize((w, h), Image.ANTIALIAS)
        for i in range(len(blending_list)):  # same background for each blend
            backgrounds.append(background.copy())

        if dontocclude:
            already_syn = []  # reset already_sin
        attempt = None
        for idx, img_data in enumerate(all_objects):
            # Load object and mask; augment it; find object position;
            loaded_data = img_data.load_object_data()
            if loaded_data is None:
                continue
            else:
                foreground, mask, orig_h, orig_w = loaded_data
            # Augmentations
            o_w, o_h = orig_w, orig_h
            if scale_augment:
                foreground, mask, o_h, o_w = augment_scale(
                    foreground, bg_h, mask, orig_h, orig_w, bg_w
                )
            if rotation_augment:
                max_degrees = MAX_DEGREES
                foreground, mask, o_h, o_w = augment_rotation(
                    foreground, bg_h, mask, max_degrees, bg_w
                )
            # Determine position
            xmin, xmax, ymin, ymax = img_data.get_annotation_from_mask()
            x, y, attempt = find_valid_object_position(
                already_syn, dontocclude, bg_h, o_h, o_w, bg_w, xmax, xmin, ymax, ymin
            )
            # Apply blending
            apply_blendings_and_paste_onto_background(
                backgrounds, blending_list, foreground, mask, x, y
            )
            # Create mask
            masks.append(
                create_full_size_and_sharpened_mask(
                    mask.copy(), ImgSize(bg_w, bg_h), ImgPosition(x, y)
                )
            )
            # Save category
            mask_category_ids.append(img_data.label_id)
            if idx >= len(objects):
                continue
        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
            continue  # could not create image yet, thus trying again
        else:
            break  # found synthesized image, thus break

    adjust_masks_for_occlusion(masks)  # remove overlay due to occlusion

    # apply final filter across whole image and save img
    for i in range(len(blending_list)):
        if blending_list[i] == "motion":
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
        backgrounds[i].save(img_files[i])

    return img_files, masks, mask_category_ids
