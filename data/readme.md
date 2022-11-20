# Data

If you want to use your own data, you can copy it into these folders or mount the folders correctly if you use Docker (
see [docker_run.sh](../scripts/docker_run.sh) for details)

Note, that you need a `split.json` file, with the following form (see e.g. [split.json](distractors/splits.json))

```json
{
  "test": [
  ],
  "train": [
    "misc/transport-ga1a289b15_1920.png",
    "misc/clock-g204ee03ae_1920.png",
    "misc/kerosene-lamp-gde06bdaab_1920.png",
    "misc/teddy-bear-g08d67e00b_1920.png",
    "misc/vw-gebf7ea815_1920.png",
    "misc/chair-g1b12b7580_1920.png"
  ],
  "validation": [
  ]
}
```

- it contains a list for each split (note, that for this mini demo, we only have `train` data)
- within each list, we have relative paths to the images
- optionally it can contain a fourth key `path` with the base path of the images (if they are not inside this folder)

## Background

- Images were downloaded from [Pixabay](https://pixabay.com/), see [sources.txt](backgrounds/sources.txt) for the links.
- For our dataset we used images
  from [SUN Database: Large-scale Scene Recognition from Abbey to Zoo](https://vision.princeton.edu/projects/2010/SUN/)
    - download complete dataset with `wget "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"`
    - unzip with `tar -xf SUN397.tar.gz`

## Distractors

Images were downloaded from [Pixabay](https://pixabay.com/), see [sources.txt](distractors/sources.txt) for the links.
Afterwards, we applied tight-cropping (see [tight_crop.py](tight_crop.py)).


## Objects

Images are photos taken by myself, where the background was removed using [rembg](https://github.com/danielgatis/rembg)
Afterwards, we applied tight-cropping (see [tight_crop.py](tight_crop.py)).
