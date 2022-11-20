import signal

import numpy as np


def init_worker():
    """
    Catch Ctrl+C signal to termiante workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def PIL2array1C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])


def PIL2array3C(img):
    """Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    """
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
