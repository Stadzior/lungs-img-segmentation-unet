from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.transform as transform
import skimage.io as io
import os

def NormalizeData(img, mask):
    #normalization
    img = img / 255
    mask = mask /255
    #tresholding
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

def CreateTrainGenerator(train_path, batch_size, target_size, image_dir, mask_dir, aug_dir, aug_parameters,
                         image_color_mode = "grayscale", mask_color_mode = "grayscale", seed = 1):
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
        img, mask = NormalizeData(img, mask)
        yield (img, mask)
        
def CreateTestGenerator(test_path, target_size, img_limit = None):
    img_filenames = list(filter(lambda img_filename: 
                                    img_filename.endswith(".png"), os.listdir(test_path)))[:img_limit]
    for img_filename in img_filenames:
        img = io.imread("{0}/{1}".format(test_path, img_filename), True)
        img = img / 255
        img = transform.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,)+img.shape)
        yield (img_filename, img)

def SaveResult(save_path, result):
    for (img_filename, img) in result:
        io.imsave("{0}/{1}".format(save_path, img_filename), img)
