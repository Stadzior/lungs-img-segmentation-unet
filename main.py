from model import *
from dataHelper import *    
from tensorboardHelper import *
from logHelper import *
from tensorflow.python.keras.callbacks import TensorBoard
from customCallbacks import AccuracyAndLossCallback
from feedType import FeedType
from keras.callbacks import ModelCheckpoint
import os

# Use pretrained weights or perform training
USE_PRETRAINED_WEIGHTS = False
PRETRAINED_WEIGHTS_FILENAME_SAVE = 'best_checkpoint_save.hdf5'
PRETRAINED_WEIGHTS_FILENAME_LOAD = 'best_checkpoint_load.hdf5'

# Feeding mode etc.
REFEED_DATA = True
FEED_TYPE = FeedType.ByRatio
OMIT_EMPTY_IMGS = True

# Used by FeedType.ByRatio
SAMPLE_COUNT = 0 # Sth less than 1 to use whatever was copied into train/test dirs
TRAIN_TO_TEST_RATIO = 0.8

# Used by FeedType.ByCounts
TRAIN_SET_COUNT = 8
TEST_SET_COUNT = 2

TARGET_SIZE = (512, 512)
EPOCH_COUNT = 2
BATCH_SIZE = 1
SOURCE_PATH = 'data/source'
TEST_PATH = 'data/test'
TRAIN_PATH = 'data/train'
RESULT_PATH = 'data/result'
IMAGE_DIR = 'image'
MASK_DIR = 'mask'
BIT_DEPTH = 8
MAX_VALUE = math.pow(2, BIT_DEPTH)-1
THRESHOLD = 0.5

AUG_ENABLED = False
AUG_DIR = 'aug'
AUG_COUNT = 100
AUG_PARAMETERS = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
   
save_path = "{0}/result_{1}".format(RESULT_PATH, datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
log_file_path = "{0}/log.txt".format(save_path)

#PrintImagesWithoutMasks('data/source/image', 'data/source/mask')
#PrintMasksWithoutImages('data/source/image', 'data/source/mask')

if (REFEED_DATA):
    ClearSets(TRAIN_PATH, TEST_PATH, IMAGE_DIR, MASK_DIR, AUG_DIR)
    train_set_count, test_set_count = FeedSets(SOURCE_PATH, TRAIN_PATH, TEST_PATH, IMAGE_DIR, MASK_DIR, FEED_TYPE, TRAIN_SET_COUNT, TEST_SET_COUNT, TRAIN_TO_TEST_RATIO, SAMPLE_COUNT, OMIT_EMPTY_IMGS)
else:
    ClearSet("{0}/{1}".format(TRAIN_PATH, AUG_DIR))
    
if not os.path.exists(save_path):
    os.makedirs(save_path)

LogParameters(log_file_path, TARGET_SIZE, EPOCH_COUNT, SAMPLE_COUNT, train_set_count,
                  BATCH_SIZE, test_set_count, BIT_DEPTH, THRESHOLD, AUG_PARAMETERS)

model = Unet((TARGET_SIZE[0], TARGET_SIZE[1], 1))
if USE_PRETRAINED_WEIGHTS:
    	model.load_weights(PRETRAINED_WEIGHTS_FILENAME_LOAD)
else:
    # Starting tensorboard server
    tensorboardServer = TensorboardHelper(save_path)
    tensorboardServer.run()
    
    # Callbacks
    tensorboardCallback = TensorBoard(save_path)
    accuracyAndLossCallback = AccuracyAndLossCallback()
    checkpointCallback = ModelCheckpoint(PRETRAINED_WEIGHTS_FILENAME_SAVE, monitor='loss',verbose=1, save_best_only=True)

    steps_per_epoch = train_set_count
    # Executing training with timestamps and measurements
    if (AUG_ENABLED):
        steps_per_epoch = steps_per_epoch * AUG_COUNT
        trainGenerator = CreateTrainGeneratorWithAugmentation(TRAIN_PATH, BATCH_SIZE, TARGET_SIZE, IMAGE_DIR, MASK_DIR, AUG_DIR, AUG_PARAMETERS, MAX_VALUE, THRESHOLD)
    else:
        trainGenerator = CreateTrainGenerator(TRAIN_PATH, IMAGE_DIR, MASK_DIR, MAX_VALUE, THRESHOLD)

    ExecuteWithLogs("Training", log_file_path, lambda _ = None: model.fit_generator(trainGenerator, steps_per_epoch, EPOCH_COUNT, callbacks = [tensorboardCallback, accuracyAndLossCallback, checkpointCallback]))   
    LogAccuracyAndLoss(log_file_path, accuracyAndLossCallback.accuracy, accuracyAndLossCallback.loss)
    os.rename(PRETRAINED_WEIGHTS_FILENAME_SAVE, PRETRAINED_WEIGHTS_FILENAME_LOAD)

# Executing testing with timestamps and measurements
testGenerator = CreateTestGenerator(TEST_PATH, TARGET_SIZE, MAX_VALUE)
result = ExecuteWithLogs("Testing", log_file_path, lambda _ = None: (model.predict_generator(testGenerator, test_set_count, verbose = 1)))
ExecuteWithLogs("Saving results", log_file_path, lambda _ = None: 
    SaveResult(TEST_PATH, save_path, result, THRESHOLD))

input("You can now go to Tensorboard url and look into the statistics (if training was performed).\nTo end session and kill Tensorboard instance input any key...")
