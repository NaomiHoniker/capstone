import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Creation Parameters
img_folder_name = "rps_images"

batch_size = 32
img_height = 180
img_width = 180
seed = 123

# Training Set Creation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=img_folder_name,
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Validation Set Creation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=img_folder_name,
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Class Name Prints (since using "inferred" labels, these should be the subdirectories)
print(train_ds.class_names)
print(val_ds.class_names)
