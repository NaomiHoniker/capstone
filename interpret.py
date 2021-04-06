import tensorflow as tf
from tensorflow import keras
import numpy as np


class SavedModel:
    def __init__(self):
        set_name = "sign_language"
        self.model = tf.keras.models.load_model('model/' + set_name + '_model')
        f = open('model/' + set_name + '_model/model_classes.txt', 'r')
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

        text_prediction = "This image is most likely " + str(self.class_labels[np.argmax(score)]) + " with a " + str(
            np.round(100 * np.max(score), 2)) + " percent accuracy."

        print(text_prediction)
        return text_prediction
