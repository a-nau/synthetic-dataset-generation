import numpy as np
import cv2


def adjust_gamma_of_image(image, gamma=1.0):
    # https://stackoverflow.com/a/41061351
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)
