import math

import numpy as np
from PIL import Image
from pyblur import LinearMotionBlur

from src.generator.utils import PIL2array1C


def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3, 5, 7, 9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes))
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:, :, i] = PIL2array1C(
            LinearMotionBlur(img[:, :, i], lineLength, lineAngle, lineType)
        )
    blurred_img = Image.fromarray(blurred_img, "RGB")
    return blurred_img


def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """
    kernelCenter = int(math.floor(kerneldim / 2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])
