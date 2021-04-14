import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from math import ceil

class SavedModel:
    def __init__(self):
        self.outputDict = {}
        self.dictIndex = 0
        self.timeDiff = 0
        self.timeDiffIndex = 0
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

        text_prediction = str(self.class_labels[np.argmax(score)]).strip('\'')

        set_output, output = self.key_value(text_prediction)

        if set_output:
            return True, output
        else:
            return False, output

    def key_value(self, classification):
        self.get_timing()
        if classification not in self.outputDict:
            self.outputDict[classification] = 1
        else:
            self.outputDict[classification] += 1
        print(self.outputDict)
        self.dictIndex += 1
        if self.timeDiffIndex != 0:
            if self.timeDiffIndex <= self.dictIndex:
                d_max = max(self.outputDict, key=self.outputDict.get)
                if self.outputDict[d_max] >= ceil(self.dictIndex * .75) and d_max != 'nothing':
                    self.dictIndex = 0
                    self.outputDict.clear()
                    return True, d_max
                self.dictIndex = 0
                self.outputDict.clear()
        return False, ""

    def get_timing(self):
        if self.dictIndex % 2 == 0:
            self.timeDiff = time.process_time()
        elif self.dictIndex % 2 == 1:
            self.timeDiff = (time.process_time() - self.timeDiff)
            cur_index = ceil(3 / self.timeDiff)
            if self.timeDiffIndex != 0:
                self.timeDiffIndex = (self.timeDiffIndex + cur_index) / 2
            else:
                self.timeDiffIndex = cur_index
            print(self.timeDiffIndex)

