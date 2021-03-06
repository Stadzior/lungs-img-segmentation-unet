import time  
import datetime
import math

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

def LogParameters(log_file_path, target_size, epoch_count, sample_count, train_set_count, test_set_count,
                  batch_size, bit_depth, threshold, aug_count, img_layer_range, aug_parameters):
    with open(log_file_path, 'a') as log_file:
        log_file.write("MODEL:\n")
        log_file.write("target size: {0}x{1}\n".format(target_size[0], target_size[1]))
        log_file.write("epoch count: {0}\n".format(epoch_count))
        log_file.write("sample count: {0}\n".format(sample_count))
        log_file.write("train set count: {0}\n".format(train_set_count))
        log_file.write("test set count: {0}\n".format(test_set_count))
        log_file.write("img layer range: ({0},{1})\n".format(img_layer_range[0], img_layer_range[1]))
        log_file.write("batch size: {0}\n".format(batch_size))
        log_file.write("bit depth: {0}\n".format(bit_depth))
        log_file.write("threshold: {0}\n".format(threshold))
        log_file.write("AUGMENTATION:\n")
        log_file.write("aug count: {0}\n".format(aug_count))
        log_file.write("rotation range: {0}\n".format(aug_parameters.get("rotation_range")))
        log_file.write("width shift range: {0}\n".format(aug_parameters.get("width_shift_range")))
        log_file.write("height shift range: {0}\n".format(aug_parameters.get("height_shift_range")))
        log_file.write("shear range: {0}\n".format(aug_parameters.get("shear_range")))
        log_file.write("zoom range: {0}\n".format(aug_parameters.get("zoom_range")))
        log_file.write("horizontal flip: {0}\n".format(aug_parameters.get("horizontal_flip")))
        log_file.write("fill mode: {0}\n".format(aug_parameters.get("fill_mode")))

def LogAccuracyAndLoss(log_file_path, accuracy, loss):
    with open(log_file_path, 'a') as log_file:
        log_file.write("Accuracy: {0},  Loss: {1}\n".format(accuracy, loss))