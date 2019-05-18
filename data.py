from keras.preprocessing.image import ImageDataGenerator

def CreateTrainGenerator(batch_size, target_size, train_path, image_dir, mask_dir, save_to_dir, aug_parameters,
                         image_color_mode = "grayscale", mask_color_mode = "grayscale", seed = 1):

    image_datagen = ImageDataGenerator(**aug_parameters)
    mask_datagen = ImageDataGenerator(**aug_parameters)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_dir],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_dir,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_dir],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_dir,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        yield (img,mask)