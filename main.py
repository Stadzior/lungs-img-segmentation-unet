from model import *
from data import *

TARGET_SIZE = (512, 512)
EPOCH_COUNT = 2
SAMPLE_COUNT = 500
BATCH_SIZE = 1
TRAIN_PATH = 'data/train'
IMAGE_DIR = 'image'
MASK_DIR = 'mask'
AUG_DIR = 'aug'
AUG_PARAMETERS = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

trainGenerator = CreateTrainGenerator(BATCH_SIZE, TARGET_SIZE, TRAIN_PATH, IMAGE_DIR,
                                      MASK_DIR, AUG_DIR, AUG_PARAMETERS)
