import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Creation parameters
import figure_creation

img_folder_name = "rps_images"

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
print(train_ds.class_names)
class_names = train_ds.class_names

print(val_ds.class_names)

# Configuring dataset for performance
# Cache images once loaded off disk during first epoch. Reduces Bottleneck
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# Begin model creation
num_classes = len(next(os.walk(img_folder_name))[1])

# Standard model creation approach
# 3 convolution blocks with max pool layer in each
# Fully connected layer with 128 units on top
model = Sequential([
    # Normalize RGB channel values for the neural net
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualization on first 9 images in the training set
figure_creation.save_first_9(train_ds, class_names)
# Save results of model accuracies
figure_creation.save_training_results(history, epochs)

# Testing Classification here, delete in full build:
hand_path = "paper1.jpg"

img = keras.preprocessing.image.load_img(
    hand_path, target_size=(180, 180)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Creating a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image is most likely '{}', with a {:.2f} percent accuracy."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)