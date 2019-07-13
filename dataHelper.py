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
    image_path = "{0}/{1}".format(train_path, image_dir)
    mask_path = "{0}/{1}".format(train_path, mask_dir)
    image_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir(image_path)))
    mask_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir(mask_path)))
    train_set_filenames = zip(image_filenames, mask_filenames)
    for (img_filename, mask_filename) in train_set_filenames:

        img = Image.open("{0}/{1}".format(image_path, img_filename), mode='r')
        img_array = np.asarray(img)
        img = ExtendToFourDims(img_array)
        img = img / max_value
        
        mask = Image.open("{0}/{1}".format(mask_path, mask_filename), mode='r')
        mask_array = np.asarray(mask)
        mask = ExtendToFourDims(mask_array)
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

def DivideAndFeedSets(source_path, train_path, test_path, image_dir, mask_dir, train_to_test_ratio, sample_count = 0, omit_empty_imgs = True):
    image_source_path = "{0}/{1}".format(source_path, image_dir)
    mask_source_path = "{0}/{1}".format(source_path, mask_dir)
    png_files = list(filter(lambda x: x.endswith(".png"), os.listdir(image_source_path)))  
    random.shuffle(png_files)
    sample_count = sample_count if sample_count > 0 else len(png_files)
    train_set, test_set = GetSets(png_files, mask_source_path, sample_count, train_to_test_ratio, omit_empty_imgs)
    print("Copying from source to train dir")
    for i, image_filename in enumerate(train_set):
        print("{0}/{1}".format(i+1, len(train_set)))
        mask_filename = GetMaskFileName(image_filename, mask_source_path)
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}/{2}".format(train_path, image_dir, image_filename))
        copyfile("{0}/{1}".format(mask_source_path, mask_filename), "{0}/{1}/{2}".format(train_path, mask_dir, mask_filename))    
    print("Copying from source to test dir")
    for i, image_filename in enumerate(test_set):        
        print("{0}/{1}".format(i+1, len(test_set)))
        mask_filename = GetMaskFileName(image_filename, mask_source_path)
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}".format(test_path, image_filename))
        copyfile("{0}/{1}".format(mask_source_path, mask_filename), "{0}/{1}".format(test_path, mask_filename))

def ExtendToFourDims(img):
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img

def IsEmptyImage(img_path):
    img = Image.open(img_path)
    return img.convert("L").getextrema()[1] == 0

def GetMaskFileName(image_filename, mask_source_path):
    return list(filter(lambda x: x.replace("_Delmon_CompleteMM", "").startswith(image_filename) and x.endswith(".png"), os.listdir(mask_source_path)))[0]

def GetSets(png_files, mask_source_path, sample_count, train_to_test_ratio, omit_empty_imgs):
    train_set_count = round(sample_count * train_to_test_ratio)
    test_set_count = sample_count - train_set_count
    train_set = []
    test_set = []
    if (omit_empty_imgs):            
        i = 0
        while i < len(png_files) and len(train_set) < train_set_count:
            image_filename = png_files[i]
            if not IsEmptyImage("{0}/{1}".format(mask_source_path, GetMaskFileName(image_filename, mask_source_path))):
                train_set.append(image_filename)
            i+=1        
        if (i >= len(png_files)):
            train_set, test_set = GetSetsWithSamplesDistributedByRatio(train_set, test_set, train_to_test_ratio)
        else:
            while i < len(png_files) and len(test_set) < test_set_count:
                image_filename = png_files[i]
                if not IsEmptyImage("{0}/{1}".format(mask_source_path, GetMaskFileName(image_filename, mask_source_path))):
                    test_set.append(image_filename)
                i+=1
            if (i >= len(png_files)):
                train_set, test_set = GetSetsWithSamplesDistributedByRatio(train_set, test_set, train_to_test_ratio)
    else:
        train_set = png_files[:train_set_count]
        test_set = png_files[train_set_count:train_set_count+train_set_count]
    return train_set, test_set
        
def GetSetsWithSamplesDistributedByRatio(train_set, test_set, train_to_test_ratio):
    samples = train_set+test_set
    train_set_count = round(len(samples) * train_to_test_ratio)
    test_set_count = len(samples) - train_set_count
    train_set = samples[:train_set_count]
    test_set = samples[train_set_count:train_set_count+train_set_count]