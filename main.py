from model import *
from dataHelper import *    
from tensorboardHelper import *
from logHelper import *
import time
import datetime
import math
from tensorflow.python.keras.callbacks import TensorBoard

TARGET_SIZE = (512, 512)
EPOCH_COUNT = 2
SAMPLE_COUNT = 300
BATCH_SIZE = 1
TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
IMAGE_DIR = 'image'
MASK_DIR = 'mask'
AUG_DIR = 'aug'
STEPS = 3
BIT_DEPTH = 8
MAX_VALUE = math.pow(2, BIT_DEPTH)-1
THRESHOLD = 0.5
AUG_PARAMETERS = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
   
def ExecuteWithLogs(func_desc, log_file_path, func_to_measure):   
    start_tmstmp = time.time()
    start_datetime = datetime.datetime.fromtimestamp(start_tmstmp).strftime("%d-%m-%Y, %H:%M:%S")    
    with open(log_file_path, 'a') as log_file:
        log_file.write("{0} started at {1}.\n".format(func_desc, start_datetime))
    result = func_to_measure()
    end_tmstmp = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end_tmstmp).strftime("%d-%m-%Y, %H:%M:%S")
    duration = end_tmstmp - start_tmstmp
    hours = math.floor(duration / 3600)
    minutes = math.floor((duration - (hours * 3600)) / 60)
    seconds = int(duration - (hours * 3600) - (minutes * 60))
    with open(log_file_path, 'a') as log_file:
        log_file.write("{0} ended at {1} and lasted {2}h {3}m {4}s.\n".format(func_desc, end_datetime, hours, minutes, seconds))
    return result

def LogParameters(log_file_path):
    with open(log_file_path, 'a') as log_file:
        log_file.write("MODEL:\n")
        log_file.write("target size: {0}x{1}\n".format(TARGET_SIZE[0], TARGET_SIZE[1]))
        log_file.write("epoch count: {0}\n".format(EPOCH_COUNT))
        log_file.write("sample count: {0}\n".format(SAMPLE_COUNT))
        log_file.write("batch size: {0}\n".format(BATCH_SIZE))
        log_file.write("steps: {0}\n".format(STEPS))
        log_file.write("bit depth: {0}\n".format(BIT_DEPTH))
        log_file.write("threshold: {0}\n".format(THRESHOLD))
        log_file.write("AUGMENTATION:\n")
        log_file.write("rotation range: {0}\n".format(AUG_PARAMETERS.get("rotation_range")))
        log_file.write("width shift range: {0}\n".format(AUG_PARAMETERS.get("width_shift_range")))
        log_file.write("height shift range: {0}\n".format(AUG_PARAMETERS.get("height_shift_range")))
        log_file.write("shear range: {0}\n".format(AUG_PARAMETERS.get("shear_range")))
        log_file.write("zoom range: {0}\n".format(AUG_PARAMETERS.get("zoom_range")))
        log_file.write("horizontal flip: {0}\n".format(AUG_PARAMETERS.get("horizontal_flip")))
        log_file.write("fill mode: {0}\n".format(AUG_PARAMETERS.get("fill_mode")))

save_path = "{0}/result_{1}".format(TEST_PATH, datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
log_file_path = "{0}/log.txt".format(save_path)
tensorboard_log_path = "{0}/tensorboard"

if not os.path.exists(save_path):
    os.makedirs(save_path)
LogParameters(log_file_path)

tensorboardServer = TensorboardHelper(tensorboard_log_path)
tensorboardServer.run()

tensorboard = TensorBoard(tensorboard_log_path)
model = Unet((TARGET_SIZE[0], TARGET_SIZE[1], 1))
#Executing training with timestamps and measurements
trainGenerator = CreateTrainGenerator(TRAIN_PATH, BATCH_SIZE, TARGET_SIZE, IMAGE_DIR, MASK_DIR, AUG_DIR,
                                    AUG_PARAMETERS, MAX_VALUE, THRESHOLD)
ExecuteWithLogs("Training", log_file_path, lambda _ = None: model.fit_generator(trainGenerator, SAMPLE_COUNT, EPOCH_COUNT, callbacks = [tensorboard]))   

#Executing testing with timestamps and measurements
testGenerator = CreateTestGenerator(TEST_PATH, TARGET_SIZE, MAX_VALUE)
result = ExecuteWithLogs("Testing", log_file_path, lambda _ = None: (model.predict_generator(testGenerator, STEPS, verbose = 1)))

#Executing result saving with timestamps and measurements
ExecuteWithLogs("Saving results", log_file_path, lambda _ = None: 
    SaveResult(TEST_PATH, save_path, result, THRESHOLD))

input("You can now go to Tensorboard url and look into the statistics.\n To end session and kill Tensorboard instance press any key...")
