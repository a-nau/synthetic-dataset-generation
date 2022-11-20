from collections import namedtuple

Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")
ImgSize = namedtuple("ImgSize", "width height")
ImgPosition = namedtuple("ImgPosition", "x y")
BaseImg = namedtuple("BaseImg", "path label")
