from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.transform as transform
import skimage.io as io
import os
import random
from shutil import copyfile
from PIL import Image

def ThresholdImage(img, threshold):    
    img[img > threshold] = 1
    img[img <= threshold] = 0
    return img

def CreateTrainGeneratorWithAugmentation(train_path, batch_size, target_size, image_dir, mask_dir, aug_dir, aug_parameters,
                         max_value, threshold, image_color_mode = "grayscale", mask_color_mode = "grayscale", seed = 1):
    #data augmentation
    image_generator = ImageDataGenerator(**aug_parameters)
    image_set = image_generator.flow_from_directory(
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
    mask_set = mask_generator.flow_from_directory(
        train_path,
        classes = [mask_dir],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = "{0}/{1}".format(train_path, aug_dir),
        save_prefix  = mask_dir,
        seed = seed)

    train_set = zip(image_set, mask_set)

    for (img, mask) in train_set:
        #normalization
        img = img / max_value
        mask = ThresholdImage(mask / max_value, threshold)        
        yield (img, mask)

def CreateTrainGenerator(train_path, image_dir, mask_dir, max_value, threshold):
    #data augmentation    
    image_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir("{0}/{1}".format(train_path, image_dir))))
    mask_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir("{0}/{1}".format(train_path, mask_dir))))
    train_set_filenames = zip(image_filenames, mask_filenames)
    for (img_filename, mask_filename) in train_set_filenames:

        img = Image.open(img_filename, mode='r')
        img_array = np.asarray(img)
        img = ExtendToFourDims(img)
        img = img / max_value
        
        mask = Image.open(mask_filename, mode='r')
        mask_array = np.asarray(mask)
        mask = ExtendToFourDims(mask)
        mask = mask / max_value
        mask = ThresholdImage(mask / max_value, threshold)      
          
        yield (img, mask)

def CreateTestGenerator(test_path, target_size, max_value):
    test_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir(test_path)))
    for filename in test_filenames:
        img = io.imread("{0}/{1}".format(test_path, filename), True)
        img = img / max_value
        img = transform.resize(img, target_size)
        yield ExtendToFourDims(img)

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
    image_source_path = "{0}/{1}".format(source_path, image_dir)
    mask_source_path = "{0}/{1}".format(source_path, mask_dir)
    png_files = list(filter(lambda x: x.endswith(".png"), os.listdir(image_source_path)))  
    random.shuffle(png_files)
    sample_count = sample_count if sample_count > 0 else len(png_files)
    train_set_count = round(sample_count * train_to_test_ratio)
    test_set_count = sample_count - train_set_count
    train_set = png_files[:train_set_count]
    test_set = png_files[-test_set_count:]
    print("Copying from source to train dir")
    for i, image_filename in enumerate(train_set):
        print ("{0}/{1}".format(i, train_set_count))
        mask_filename = list(filter(lambda x: x.replace("_Delmon_CompleteMM", "").startswith(image_filename) and x.endswith(".png"), os.listdir(mask_source_path)))[0]
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}/{2}".format(train_path, image_dir, image_filename))
        copyfile("{0}/{1}".format(mask_source_path, mask_filename), "{0}/{1}/{2}".format(train_path, mask_dir, mask_filename))    
    print("Copying from source to test dir")
    for i, image_filename in enumerate(test_set):        
        print ("{0}/{1}".format(i, test_set_count))
        mask_filename = list(filter(lambda x: x.replace("_Delmon_CompleteMM", "").startswith(image_filename) and x.endswith(".png"), os.listdir(mask_source_path)))[0]
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}".format(test_path, image_filename))
        copyfile("{0}/{1}".format(mask_source_path, mask_filename), "{0}/{1}".format(test_path, mask_filename))

def ExtendToFourDims(img):
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img