from model import *
from dataHelper import *    
from tensorboardHelper import *
from logHelper import *
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
   
save_path = "{0}/result_{1}".format(TEST_PATH, datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
log_file_path = "{0}/log.txt".format(save_path)
tensorboard_log_path = "{0}/tensorboard"

if not os.path.exists(save_path):
    os.makedirs(save_path)

LogParameters(log_file_path, TARGET_SIZE, EPOCH_COUNT, SAMPLE_COUNT,
                  BATCH_SIZE, STEPS, BIT_DEPTH, THRESHOLD, AUG_PARAMETERS)

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
