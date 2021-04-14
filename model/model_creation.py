import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Creation parameters
from model import figure_creation

image_set_name = "sign_language"

img_folder_name = "../" + image_set_name + "_image_set"

batch_size = 32
img_height = 180
img_width = 180
seed = 123

# Training set creation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=img_folder_name,
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Validation set creation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=img_folder_name,
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Class name prints (since using "inferred" labels, these should be the subdirectories)
class_names = train_ds.class_names

# Configuring dataset for performance
# Cache images once loaded off disk during first epoch. Reduces Bottleneck
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# Begin model creation
num_classes = len(next(os.walk(img_folder_name))[1])

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomContrast(0.15),
    ]
)

figure_creation.save_augmented_images(train_ds, data_augmentation)

# Standard model creation approach
# 3 convolution blocks with max pool layer in each
# Fully connected layer with 128 units on top
model = Sequential([
    # Normalize RGB channel values for the neural net
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.15),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualization on first 9 images in the training set
figure_creation.save_first_9(train_ds, class_names)
# Save results of model accuracies
figure_creation.save_training_results(history, epochs)

# Save the model
model.save(image_set_name + '_model')
f = open(image_set_name + "_model/model_classes.txt", 'w')
f.write(str(class_names))
f.close()
