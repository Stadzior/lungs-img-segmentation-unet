from model import *
from dataHelper import *    
from tensorboardHelper import *
from logHelper import *
from tensorflow.python.keras.callbacks import TensorBoard
from customCallbacks import AccuracyAndLossCallback

TARGET_SIZE = (512, 512)
EPOCH_COUNT = 1
SAMPLE_COUNT = 10
TRAIN_TO_TEST_RATIO = 0.8
BATCH_SIZE = 1
SOURCE_PATH = 'data/source'
TEST_PATH = 'data/test'
TRAIN_PATH = 'data/train'
RESULT_PATH = 'data/result'
IMAGE_DIR = 'image'
MASK_DIR = 'mask'
STEPS = 3
BIT_DEPTH = 8
MAX_VALUE = math.pow(2, BIT_DEPTH)-1
THRESHOLD = 0.5
AUG_DIR = 'aug'
AUG_COUNT = 50
AUG_PARAMETERS = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
AUG_ENABLED = True
   
save_path = "{0}/result_{1}".format(RESULT_PATH, datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
log_file_path = "{0}/log.txt".format(save_path)

# ClearSets(TRAIN_PATH, TEST_PATH, IMAGE_DIR, MASK_DIR, AUG_DIR)
# DivideAndFeedSets(SOURCE_PATH, TRAIN_PATH, TEST_PATH, IMAGE_DIR, MASK_DIR, TRAIN_TO_TEST_RATIO, SAMPLE_COUNT)

if not os.path.exists(save_path):
    os.makedirs(save_path)

LogParameters(log_file_path, TARGET_SIZE, EPOCH_COUNT, SAMPLE_COUNT, TRAIN_TO_TEST_RATIO,
                  BATCH_SIZE, STEPS, BIT_DEPTH, THRESHOLD, AUG_PARAMETERS)

tensorboardServer = TensorboardHelper(save_path)
tensorboardServer.run()

tensorboardCallback = TensorBoard(save_path)
accuracyAndLossCallback = AccuracyAndLossCallback()
model = Unet((TARGET_SIZE[0], TARGET_SIZE[1], 1))

steps_per_epoch = round(SAMPLE_COUNT * TRAIN_TO_TEST_RATIO)
#Executing training with timestamps and measurements
if (AUG_ENABLED):
    steps_per_epoch = steps_per_epoch * AUG_COUNT
    trainGenerator = CreateTrainGeneratorWithAugmentation(TRAIN_PATH, BATCH_SIZE, TARGET_SIZE, IMAGE_DIR, MASK_DIR, AUG_DIR, AUG_PARAMETERS, MAX_VALUE, THRESHOLD)
else:
    trainGenerator = CreateTrainGenerator(TRAIN_PATH, IMAGE_DIR, MASK_DIR, MAX_VALUE, THRESHOLD)

ExecuteWithLogs("Training", log_file_path, lambda _ = None: model.fit_generator(trainGenerator, steps_per_epoch, EPOCH_COUNT, callbacks = [tensorboardCallback, accuracyAndLossCallback]))   

#Executing testing with timestamps and measurements
testGenerator = CreateTestGenerator(TEST_PATH, TARGET_SIZE, MAX_VALUE)
result = ExecuteWithLogs("Testing", log_file_path, lambda _ = None: (model.predict_generator(testGenerator, STEPS, verbose = 1)))
ExecuteWithLogs("Saving results", log_file_path, lambda _ = None: 
    SaveResult(TEST_PATH, save_path, result, THRESHOLD))

LogAccuracyAndLoss(log_file_path, accuracyAndLossCallback.accuracy, accuracyAndLossCallback.loss)

input("You can now go to Tensorboard url and look into the statistics.\n To end session and kill Tensorboard instance input any key...")
