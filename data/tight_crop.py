import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from PIL import Image

if __name__ == "__main__":
    image_paths = list((ROOT / "data" / "objects").rglob("*.png")) + list(
        (ROOT / "data" / "distractors").rglob("*.png")
    )

    for image_path in image_paths:
        print(f"Cropping: {image_path}")
        image = Image.open(image_path.as_posix())
        cropped = image.crop(image.getbbox())
        cropped.save(image_path.as_posix())
