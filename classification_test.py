import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Creation parameters
import figure_creation


def classification_test(model):
    hand_path = "/hand_test/paper/paper1"

    img = keras.preprocessing.image.load_img(
        hand_path, target_size=(180, 180)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Creating a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "I'm {} percent sure this is a {:.2f}."
        .format(100 * np.max(score), class_names[np.argmax(score)])
    )
