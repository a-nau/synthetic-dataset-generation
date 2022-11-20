import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.generator.handler import generate_synthetic_dataset


class TestDatasetGeneration(unittest.TestCase):
    def test_dataset_generation(self):
        dataset_name = "test_dataset"  # Give your dataset a name
        output_dir = (ROOT / "data" / dataset_name).resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir.as_posix())
        output_dir.mkdir(parents=True)
        # Adjust paths here if you are not using Docker
        distractor_json = ROOT / "data/distractors/splits.json"
        object_json = ROOT / "data/objects/splits.json"
        background_json = ROOT / "data/backgrounds/splits.json"
        generate_synthetic_dataset(
            output_dir=str(output_dir),
            object_json=str(object_json),
            distractor_json=str(distractor_json),
            background_json=str(background_json),
            number_of_images={
                "train": 1,
                "validation": 0,
                "test": 0,
            },  # multiplied by blending methods,
            dontocclude=True,  # enable occlusion checking of objects
            rotation=True,  # enable random rotation of objects
            scale=True,  # enable random scaling of objects
            multithreading=False,  # enable multithreading for faster dataset generation
        )
