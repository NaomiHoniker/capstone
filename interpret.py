import tensorflow as tf
from tensorflow import keras
import numpy as np


class RpsSavedModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/rps_model')
        f = open('model/rps_model/model_classes.txt', 'r')
        string = f.read().strip('[').strip(']')
        self.class_labels = string.split(', ')
        print(self.class_labels)

    def interpret(self):
        img = keras.preprocessing.image.load_img(
            'img_to_interpret.png', target_size=(180, 180)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Creating a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image is most likely '{}', with a {:.2f} percent accuracy."
                .format(self.class_labels[np.argmax(score)], 100 * np.max(score))
        )
