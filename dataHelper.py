from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.transform as transform
import skimage.io as io
import os
import random
from shutil import copyfile

def ThresholdImage(img, threshold):    
    img[img > threshold] = 1
    img[img <= threshold] = 0
    return img

def CreateTrainGenerator(train_path, batch_size, target_size, image_dir, mask_dir, aug_dir, aug_parameters,
                         max_value, threshold, image_color_mode = "grayscale", mask_color_mode = "grayscale", seed = 1):
    #data augmentation
    image_generator = ImageDataGenerator(**aug_parameters)
    image_generator = image_generator.flow_from_directory(
        train_path,
        classes = [image_dir],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = "{0}/{1}".format(train_path, aug_dir),
        save_prefix  = image_dir,
        seed = seed)

    mask_generator = ImageDataGenerator(**aug_parameters)
    mask_generator = mask_generator.flow_from_directory(
        train_path,
        classes = [mask_dir],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = "{0}/{1}".format(train_path, aug_dir),
        save_prefix  = mask_dir,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        #normalization
        img = img / max_value
        mask = ThresholdImage(mask / max_value, threshold)        
        yield (img, mask)

def CreateTestGenerator(test_path, target_size, max_value):
    test_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir(test_path)))
    for filename in test_filenames:
        img = io.imread("{0}/{1}".format(test_path, filename), True)
        img = img / max_value
        img = transform.resize(img, target_size)
        #extend to 4 dims
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,)+img.shape)
        yield img

def SaveResult(test_path, save_path, result, threshold):    
    png_files = list(filter(lambda x: x.endswith(".png"), os.listdir(test_path)))
    for i, img in enumerate(result):
        img = ThresholdImage(img[:,:,0], threshold)
        io.imsave("{0}/{1}".format(save_path, png_files[i]), img)

def ClearSets(train_path, test_path, image_dir, mask_dir, aug_dir):
    for path in [test_path, "{0}/{1}".format(train_path, image_dir), "{0}/{1}".format(train_path, mask_dir), "{0}/{1}".format(train_path, aug_dir)]: 
        png_files = list(filter(lambda x: x.endswith(".png"), os.listdir(path)))
        for file in png_files:
            os.remove("{0}/{1}".format(path, file))

def DivideAndFeedSets(source_path, train_path, test_path, image_dir, mask_dir, train_to_test_ratio, sample_count = 0):
    png_files = list(filter(lambda x: x.endswith(".png") and not "MM" in x, os.listdir(source_path)))
    train_set_count = round(len(png_files)*train_to_test_ratio)
    test_set_count = len(png_files) - train_set_count
    random.shuffle(png_files)
    train_set = png_files[:train_set_count]
    train_set = train_set[:sample_count] if sample_count > 0 else train_set
    test_set = png_files[test_set_count:]
    test_set = test_set[:sample_count] if sample_count > 0 else test_set
    for train_sample in train_set:
        image_filename = train_sample
        mask_filename = list(filter(lambda x: x.startswith(image_filename) and "MM" in x and x.endswith(".png"), os.listdir(source_path)))
        copyfile("{0}/{1}".format(source_path, image_filename), "{0}/{1}/{2}".format(train_path, image_dir, image_filename))
        copyfile("{0}/{1}".format(source_path, mask_filename), "{0}/{1}/{2}".format(train_path, mask_dir, mask_filename))
    for test_sample in test_set:
        image_filename = test_sample
        mask_filename = list(filter(lambda x: x.startswith(image_filename) and "MM" in x and x.endswith(".png"), os.listdir(source_path)))
        copyfile("{0}/{1}".format(source_path, image_filename), "{0}/{1}/{2}".format(test_path, image_dir, image_filename))
        copyfile("{0}/{1}".format(source_path, mask_filename), "{0}/{1}/{2}".format(test_path, mask_dir, mask_filename))
