from model import *
from data import *    
import time
import datetime
import math

TARGET_SIZE = (512, 512)
EPOCH_COUNT = 2
SAMPLE_COUNT = 500
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
   
def ExecuteWithTimestamps(func_desc, func_to_measure):   
    start_tmstmp = time.time()
    start_datetime = datetime.datetime.fromtimestamp(start_tmstmp).strftime("%d-%m-%Y, %H:%M:%S")
    print("{0} started at {1}".format(func_desc, start_datetime))
    result = func_to_measure()
    end_tmstmp = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end_tmstmp).strftime("%d-%m-%Y, %H:%M:%S")
    duration = end_tmstmp - start_tmstmp
    hours = math.floor(duration / 3600)
    minutes = math.floor((duration - (hours * 3600)) / 60)
    seconds = int(duration - (hours * 3600) - (minutes * 60))
    print("{0} ended at {1} and lasted {2}h {3}m {4}s".format(func_desc, end_datetime, hours, minutes, seconds))
    return result

model = Unet((TARGET_SIZE[0], TARGET_SIZE[1], 1))

#Executing training with timestamps and measurements
trainGenerator = CreateTrainGenerator(TRAIN_PATH, BATCH_SIZE, TARGET_SIZE, IMAGE_DIR, MASK_DIR, AUG_DIR,
                                      AUG_PARAMETERS, MAX_VALUE, THRESHOLD)
ExecuteWithTimestamps("Training", lambda _ = None: model.fit_generator(trainGenerator, SAMPLE_COUNT, EPOCH_COUNT))   

#Executing testing with timestamps and measurements
testGenerator = CreateTestGenerator(TEST_PATH, TARGET_SIZE, MAX_VALUE)
result = ExecuteWithTimestamps("Testing", lambda _ = None: (model.predict_generator(testGenerator, STEPS, verbose = 1)))

#Executing result saving with timestamps and measurements
save_path = "{0}/result_{1}".format(TEST_PATH, datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
ExecuteWithTimestamps("Saving results", lambda _ = None: 
    SaveResult(TEST_PATH, save_path, result, THRESHOLD))