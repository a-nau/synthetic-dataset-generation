from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from src.config import INVERTED_MASK, MINFILTER_SIZE
from src.config import OBJECT_CATEGORIES


class BaseImgData:
    def __init__(self, img_path: Path, label: str):
        self.img_path = img_path
        self.label = label
        self.label_id = [f["id"] for f in OBJECT_CATEGORIES if label == f["name"]][0]
        self.load_complementary_data()

    def __str__(self):
        return f"Label {self.label} from {self.img_path}"

    def load_complementary_data(self):
        raise NotImplementedError("Should be implemented by subclass")

    def get_mask(self, opencv=True):
        raise NotImplementedError("Should be implemented by subclass")

    def get_image(self, opencv=True):
        if opencv:
            img = cv2.imread(self.img_path.as_posix())
        else:
            img = Image.open(self.img_path.as_posix())

        # check_tensor(to_numpy_image(img), 'h w 3')
        return img

    def get_annotation_from_mask(self, scale=1.0):
        """Given a mask file and scale, return the bounding box annotations

        Args:
        Returns:
            tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
        """
        mask = self.get_mask(opencv=True)
        if mask is not None:
            if INVERTED_MASK:
                mask = 255 - mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if len(np.where(rows)[0]) > 0:
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                return (
                    int(scale * xmin),
                    int(scale * xmax),
                    int(scale * ymin),
                    int(scale * ymax),
                )
            else:
                return -1, -1, -1, -1
        else:
            print("Mask not found. Using empty mask instead.")
            return -1, -1, -1, -1

    def load_object_data(self):
        foreground = self.get_image(opencv=False)
        if foreground is None:
            return None
        xmin, xmax, ymin, ymax = self.get_annotation_from_mask()
        foreground = foreground.crop((xmin, ymin, xmax, ymax))
        orig_w, orig_h = foreground.size
        mask = self.get_mask(opencv=False)
        if mask is None:
            return None
        mask = mask.crop((xmin, ymin, xmax, ymax))
        return foreground, mask, orig_h, orig_w


class ImgDataRGBA(BaseImgData):
    def __init__(self, img_path: Path, label):
        super().__init__(img_path, label)

    def load_complementary_data(self):
        pass

    def get_mask(self, opencv=False):
        with open(self.img_path.as_posix(), "rb") as f:
            image = Image.open(f)
            if image.mode == "RGBA":
                mask = image.split()[3].filter(
                    ImageFilter.MinFilter(MINFILTER_SIZE)
                )  # MinFilter better than threshold
                if opencv:
                    mask = np.asarray(mask).astype(np.uint8)
            else:
                mask = None
        return mask

    def get_image(self, opencv=False):
        with open(self.img_path.as_posix(), "rb") as f:
            img = Image.open(f)
            if opencv:
                new_img = np.asarray(img).astype(np.uint8)[:, :, :3]
                # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            else:
                if img.mode == "RGBA":
                    new_img = Image.new(
                        "RGB", img.size, (255, 255, 255)
                    )  # white background
                    new_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                else:
                    print(f"No RGBA channel found for {self.img_path}")
                    return None
        return new_img
