import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from math import ceil
from abc import ABC, abstractmethod

ACCURACY_TO_INTERPRET = .75
SECONDS_TO_INTERPRET = 2


class SavedModel(ABC):
    def __init__(self):
        self.output_dict = {}
        self.dict_index = 0
        self.end_time = 0
        self.output = ""

    def interpret(self):
        img = keras.preprocessing.image.load_img(
            'img_to_interpret.png', target_size=(180, 180)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Creating a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        text_prediction = str(self.class_labels[np.argmax(score)]).strip('\'')
        self.key_value(text_prediction)

    def key_value(self, classification):
        if classification not in self.output_dict:
            self.output_dict[classification] = 1
        else:
            self.output_dict[classification] += 1
        print(self.output_dict)
        self.dict_index += 1
        if self.dict_index == 1:
            self.end_time = (time.time() + SECONDS_TO_INTERPRET)
        else:
            if time.time() >= self.end_time:
                d_max = max(self.output_dict, key=self.output_dict.get)
                if self.output_dict[d_max] >= ceil(self.dict_index * ACCURACY_TO_INTERPRET) and d_max != 'nothing':
                    self.output_to_screen(d_max)
                self.dict_index = 0
                self.output_dict.clear()

    @abstractmethod
    def output_to_screen(self, new_addition):
        pass

    @abstractmethod
    def keys(self, key):
        pass


class SLType(SavedModel):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model("model/sign_language_model")
        f = open("model/sign_language_model/model_classes.txt", 'r')
        self.class_labels = (f.read().strip('[').strip(']')).split(', ')
        f.close()

    def output_to_screen(self, new_addition):
        self.output += new_addition
        self.output.replace("_", " ")
        return self.output

    def keys(self, key):
        if key == ord(' '):
            self.output += "_"

        if key == ord('\b'):
            self.output = self.output[:-1]


class RPSType(SavedModel):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model("model/rps_model")
        f = open("model/rps_model/model_classes.txt", 'r')
        self.class_labels = (f.read().strip('[').strip(']')).split(', ')
        f.close()

    def output_to_screen(self, new_addition):
        self.output = new_addition
        return self.output

    def keys(self, key):
        if key == ord('\b'):
            self.output = self.output[:-1]
