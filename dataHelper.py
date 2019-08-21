from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.transform as transform
import skimage.io as io
import os
import random
from shutil import copyfile
from PIL import Image
from feedType import FeedType
from transparencyLevel import TransparencyLevel
import re

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
        ClearSet(path)

def DeleteLayers(path, startIndex, endIndex):
    png_files_to_delete = list(filter(lambda x: x.endswith(".png") and int(x[x.rfind('_')+1:-4]) > startIndex and int(x[x.rfind('_')+1:-4]) < endIndex, os.listdir(path)))
    for file in png_files_to_delete:
        os.remove('{0}/{1}'.format(path, file))

def PrintImagesWithoutMasks(image_path, mask_path):    
    image_files = list(filter(lambda x: x.endswith(".png"), os.listdir(image_path)))
    images_without_masks = []
    for i, image in enumerate(image_files):
        try:
            print("{0}/{1}".format(i, len(image_files)))
            mask = GetMaskFileName(image, mask_path)
        except IndexError:
            images_without_masks.append(image)
    for image in images_without_masks:
        print(image)

def PrintMasksWithoutImages(image_path, mask_path):
    mask_files = list(filter(lambda x: x.endswith(".png"), os.listdir(mask_path)))
    masks_without_images = []
    for i, mask in enumerate(mask_files):
        try:
            print("{0}/{1}".format(i, len(mask_files)))
            image = mask.replace("_Delmon_CompleteMM", "")
            list(filter(lambda x: x == image, os.listdir(image_path)))[0]            
        except IndexError:
            masks_without_images.append(mask)
    for mask in masks_without_images:
        print(mask)

def ClearSet(path):
    png_files = list(filter(lambda x: x.endswith(".png"), os.listdir(path)))
    for file in png_files:
        os.remove("{0}/{1}".format(path, file))

def FeedSets(source_path, train_path, test_path, image_dir, mask_dir, feed_type, train_set_count = 0, test_set_count = 0, train_to_test_ratio = 0, sample_count = 0, per_raw = True, img_layer_range  = (0, 468)):
    image_source_path = "{0}/{1}".format(source_path, image_dir)
    mask_source_path = "{0}/{1}".format(source_path, mask_dir)
    print("Working on scan layers from {0} to {1}.".format(img_layer_range[0], img_layer_range[1]))
    extractIndexFromFileName = lambda x: (int(re.search("_[\d]+.png", x).group(0).replace("_", "").replace(".png", "")))
    png_files = list(filter(lambda x: x.endswith(".png") and img_layer_range[0] <= extractIndexFromFileName(x) <= img_layer_range[1], os.listdir(image_source_path)))  
    train_set, test_set = GetSets(png_files, feed_type, sample_count, train_to_test_ratio,
                                  train_set_count, test_set_count, per_raw)    
    print("Copying from source to train dir")
    for i, image_filename in enumerate(train_set):
        print("{0}/{1}".format(i+1, len(train_set)))
        mask_filename = GetMaskFileName(image_filename, mask_source_path)
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}/{2}".format(train_path, image_dir, image_filename))
        copyfile("{0}/{1}".format(mask_source_path, mask_filename), "{0}/{1}/{2}".format(train_path, mask_dir, mask_filename))    
    print("Copying from source to test dir")
    for i, image_filename in enumerate(test_set):        
        print("{0}/{1}".format(i+1, len(test_set)))
        copyfile("{0}/{1}".format(image_source_path, image_filename), "{0}/{1}".format(test_path, image_filename))
    return len(train_set), len(test_set)

def ExtendToFourDims(img):
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img

def IsEmptyImage(img_path):
    img = Image.open(img_path)
    return img.convert("L").getextrema()[1] == 0

def GetMaskFileName(image_filename, mask_source_path):
    return list(filter(lambda x: x.replace("_Delmon_CompleteMM", "").startswith(image_filename) and x.endswith(".png"), os.listdir(mask_source_path)))[0]

def GetImageFileName(mask_filename):
    return mask_filename.replace("_Delmon_CompleteMM", "")  
  
def GetSets(png_files, feed_type, sample_count, train_to_test_ratio, train_set_count, test_set_count, per_raw): 
    sample_count_specified = sample_count > 0
    sample_count = sample_count if sample_count_specified else len(GetRawFileNames()) if per_raw else len(png_files)    
    if (feed_type == FeedType.ByRatio):
        train_set_count = round(sample_count * train_to_test_ratio)
        test_set_count = sample_count - train_set_count
    if (per_raw):
        low_raw_files, medium_raw_files, high_raw_files = (GetRawFileNamesByTransparencyLevel(TransparencyLevel.Low),
                                                GetRawFileNamesByTransparencyLevel(TransparencyLevel.Medium),
                                                GetRawFileNamesByTransparencyLevel(TransparencyLevel.High))
        random.shuffle(low_raw_files)
        random.shuffle(medium_raw_files)
        random.shuffle(high_raw_files)

        if (feed_type == FeedType.ByCount or sample_count_specified):
            train_low_raw_files = low_raw_files[:round(train_set_count / 3)]             
            train_medium_raw_files = medium_raw_files[:round(train_set_count / 3)]     
            train_high_raw_files = high_raw_files[:train_set_count - len(train_low_raw_files) - len(train_medium_raw_files)]
            test_low_raw_files = low_raw_files[len(train_low_raw_files):len(train_low_raw_files) + round(test_set_count / 3)]             
            test_medium_raw_files = medium_raw_files[len(train_medium_raw_files):len(train_medium_raw_files) + round(test_set_count / 3)]     
            test_high_raw_files = high_raw_files[len(train_high_raw_files):len(train_high_raw_files) + test_set_count - len(test_low_raw_files) - len(test_medium_raw_files)]
        else:                
            train_low_raw_files = low_raw_files[:round(len(low_raw_files) * train_to_test_ratio)]
            train_medium_raw_files = medium_raw_files[:round(len(medium_raw_files) * train_to_test_ratio)]
            train_high_raw_files = high_raw_files[:round(len(high_raw_files) * train_to_test_ratio)]
            test_low_raw_files = low_raw_files[len(train_low_raw_files):]
            test_medium_raw_files = medium_raw_files[len(train_medium_raw_files):]
            test_high_raw_files = high_raw_files[len(train_high_raw_files):]
        print("Train raw files: {0}".format(train_set_count))
        print("Low transparency: {0}".format(len(train_low_raw_files)))
        print("Medium transparency: {0}".format(len(train_medium_raw_files)))
        print("High transparency: {0}".format(len(train_high_raw_files)))
        print("Test raw files: {0}".format(test_set_count))
        print("Low transparency: {0}".format(len(test_low_raw_files)))
        print("Medium transparency: {0}".format(len(test_medium_raw_files)))
        print("High transparency: {0}".format(len(test_high_raw_files)))
        train_raw_files = train_low_raw_files + train_medium_raw_files + train_high_raw_files
        test_raw_files = test_low_raw_files + test_medium_raw_files + test_high_raw_files
        train_set = []
        test_set = []
        for png_file in png_files:
            for train_raw_file in train_raw_files:
                if (train_raw_file in png_file):
                    train_set.append(png_file)
                    break
            if (png_file not in train_set):
                for test_raw_file in test_raw_files:
                    if (test_raw_file in png_file):
                        test_set.append(png_file)
                        break
    else:
        random.shuffle(png_files)
        train_set = png_files[:train_set_count]
        test_set = png_files[train_set_count:train_set_count+test_set_count]
    return train_set, test_set

def DeleteEmptyImgs(source_path, image_dir, mask_dir):
    print("Remove empty imgs started:")
    image_source_path = "{0}/{1}".format(source_path, image_dir)
    mask_source_path = "{0}/{1}".format(source_path, mask_dir)
    mask_filenames = list(filter(lambda x: x.endswith(".png"), os.listdir(mask_source_path)))
    for i,mask_filename in enumerate(mask_filenames):  
        mask_path = "{0}/{1}".format(mask_source_path, mask_filename)     
        if IsEmptyImage(mask_path):
            os.remove(mask_path)
            image_filename = GetImageFileName(mask_filename)
            image_path = "{0}/{1}".format(image_source_path, image_filename)    
            os.remove(image_path)            
            print("{0}/{1} {2} REMOVED".format(i+1, len(mask_filenames), mask_filename))
        print("{0}/{1} {2}".format(i+1, len(mask_filenames), mask_filename))

def GetRawFileNames():
    return ["inspi16__1.0__B31f_3", "inspi15__1.0__B31f_11", "inspi14__1.0__B31f_19",
            "inspi13__1.0__B31f_27", "expi16__1.0__B31f_7", "expi15__1.0__B31f_15",
            "expi14__1.0__B31f_23", "expi13__1.0__B31f_31", "expi12__1.0__B31f_35",
            "inspi8__1.0__B31f_98", "inspi7__1.0__B31f_90", "inspi6__1.0__B31f_82",
            "inspi5__1.0__B31f_74", "inspi4__1.0__B31f_66", "inspi3__1.0__B31f_58", 
            "expi6__1.0__B31f_78", "expi5__1.0__B31f_70", "expi4__1.0__B31f_62",
            "expi3__1.0__B31f_54", "expi2__1.0__B31f_46", "inspi11__1.0__B31f_1022",
            "inspi10__1.0__B31f_1014", "inspi9__1.0__B31f_1006", "inspi1__1.0__B31f_42",
            "expi11__1.0__B31f_1018", "expi10__1.0__B31f_1010", "expi9__1.0__B31f_1002",
            "expi8__1.0__B31f_94", "expi7__1.0__B31f_86", "expi1__1.0__B31f_38"]

def GetRawFileNamesByTransparencyLevel(transparency_level):
    if (transparency_level == TransparencyLevel.Low):
        return ["inspi16__1.0__B31f_3", "inspi15__1.0__B31f_11", "inspi14__1.0__B31f_19",
                "inspi13__1.0__B31f_27", "expi16__1.0__B31f_7", "expi15__1.0__B31f_15",
                "expi14__1.0__B31f_23", "expi13__1.0__B31f_31", "expi12__1.0__B31f_35"]
    elif (transparency_level == TransparencyLevel.Medium):
        return ["inspi8__1.0__B31f_98", "inspi7__1.0__B31f_90", "inspi6__1.0__B31f_82", "inspi5__1.0__B31f_74",
                "inspi4__1.0__B31f_66", "inspi3__1.0__B31f_58", "expi6__1.0__B31f_78", "expi5__1.0__B31f_70",
                "expi4__1.0__B31f_62", "expi3__1.0__B31f_54", "expi2__1.0__B31f_46"]
    elif (transparency_level == TransparencyLevel.High):
        return ["inspi11__1.0__B31f_1022", "inspi10__1.0__B31f_1014", "inspi9__1.0__B31f_1006",
                "inspi1__1.0__B31f_42", "expi11__1.0__B31f_1018", "expi10__1.0__B31f_1010",
                "expi9__1.0__B31f_1002", "expi8__1.0__B31f_94", "expi7__1.0__B31f_86", "expi1__1.0__B31f_38"]

def CalculateAutoAugCount(img_layer_range):
    return 10

def CalculateAutoRanges(img_layer_range):
    