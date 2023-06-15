from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def train_image_gen(image_tile_dir, mask_tile_dir, batch_size=16, seed=42):
        datagen = ImageDataGenerator(
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        validation_split=0.2)

        train_image_generator = datagen.flow_from_directory(
                        directory=image_tile_dir,
                        color_mode='rgb',
                        target_size = (128, 128),
                        class_mode=None,
                        shuffle=True,
                        seed=42,
                        subset='training')

        train_mask_generator = datagen.flow_from_directory(
                        directory=mask_tile_dir,
                        color_mode='grayscale',
                        target_size = (128, 128),
                        class_mode=None,
                        shuffle=True,
                        seed=42,
                        subset='training')
        
        train_generator = zip(train_image_generator, train_mask_generator)
        for (img, mask) in train_generator:
            mask = to_categorical(mask, num_classes=2)
            yield (img, mask)
                
            
            
    def val_image_gen(image_tile_dir, mask_tile_dir, batch_size=16, seed=42):
        datagen = ImageDataGenerator(
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        validation_split=0.2)
        
        val_image_generator = datagen.flow_from_directory(
                        directory=image_tile_dir,
                        color_mode='rgb',
                        target_size = (128, 128),
                        class_mode=None,
                        shuffle=True,
                        seed=42,
                        subset='validation')

        val_mask_generator = datagen.flow_from_directory(
                        directory=mask_tile_dir,
                        color_mode='grayscale',
                        target_size = (128, 128),
                        class_mode=None,
                        shuffle=True,
                        seed=42,
                        subset='validation')

        val_generator = zip(val_image_generator, val_mask_generator)
        for (img, mask) in val_generator:
            mask = to_categorical(mask, num_classes=2)
            yield (img, mask)

        
    
    