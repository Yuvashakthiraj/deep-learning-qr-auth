"""
Image preprocessing module for the deep learning model.
"""
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array


def create_data_generators(train_dir, validation_dir, batch_size=32, target_size=(224, 224)):
    """
    Create data generators for training and validation.
    
    Args:
        train_dir: Directory containing training data
        validation_dir: Directory containing validation data
        batch_size: Batch size for training
        target_size: Target size for images
    
    Returns:
        train_generator, validation_generator
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, validation_generator


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
