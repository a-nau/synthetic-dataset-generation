# Parameters for generator
NUMBER_OF_WORKERS = 20
BLENDING_LIST = [
    "gaussian",
    # "poisson",  # takes a lot of time and results are not that good
    # "poisson-fast",  # only with Docker GPU
    "none",
    # "box",
    "motion",
    # "mixed",
    # "illumination",
    # "gamma_correction",
]

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 4
MIN_NO_OF_DISTRACTOR_OBJECTS = 2
MAX_NO_OF_DISTRACTOR_OBJECTS = 4
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
MIN_SCALE = 0.15  # min scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_SCALE = 0.4  # max scale for scale augmentation (maximum extend in each direction, 1=same size as image)
MAX_UPSCALING = 1.2  # increase size of foreground by max this
MAX_DEGREES = 30  # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = (
    0.25  # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
)
MAX_ALLOWED_IOU = 0.5  # IOU > MAX_ALLOWED_IOU is considered an occlusion, need dontocclude=True

# Parameters for image loading
MINFILTER_SIZE = 3

# Other
OBJECT_CATEGORIES = [
    {"id": 0, "name": "box"},
    {"id": 2, "name": "distractor"},
]  # note: distractor needs to be second position
IGNORE_LABELS = [OBJECT_CATEGORIES[1]["id"]]  # list of category ID for which no annotations will be generated
INVERTED_MASK = False  # Set to true if white pixels represent background
SUPPORTED_IMG_FILE_TYPES = (".jpg", "jpeg", ".png", ".gif")
