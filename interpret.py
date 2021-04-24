import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from math import ceil
from abc import ABC, abstractmethod

# Constants
# Accuracy necessary within interpretation time-frame to rewrite screen
ACCURACY_TO_INTERPRET = .75
# Seconds until next interpretation
SECONDS_TO_INTERPRET = 2


class SavedModel(ABC):
    """Default model base-class, should never be implemented (abstract)"""
    def __init__(self):
        # Dictionary of each frame interpretation within time-frame
        self.output_dict = {}
        # Number of dictionary inputs within interpretation time-frame
        self.dict_index = 0
        # Time until current interpretation time-frame ends
        self.end_time = 0
        # Output to screen
        self.output = ""

    def interpret(self):
        """Interpret current interest frame (stored on drive via 'img_to_interpret.png')"""
        img = keras.preprocessing.image.load_img(
            'img_to_interpret.png', target_size=(180, 180)
        )
        # Format image to interpretable array
        img_array = keras.preprocessing.image.img_to_array(img)
        # Creating a batch
        img_array = tf.expand_dims(img_array, 0)

        # Predict and score
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Get text prediction of classification and format to put into output_dict
        text_prediction = str(self.class_labels[np.argmax(score)]).strip('\'')
        self.key_value(text_prediction)

    def key_value(self, classification):
        """Populate the output_dict dictionary to make accurate interpretations

        Variable(s):
            classification: New classification determined by interpret function using 'img_to_interpret.png'
        """
        if classification not in self.output_dict:
            # Initialize key in dictionary
            self.output_dict[classification] = 1
        else:
            # Increase value of key
            self.output_dict[classification] += 1
        print(self.output_dict)
        self.dict_index += 1
        # For first interpretation
        if self.dict_index == 1:
            # Set end_time of interpretation time-frame
            self.end_time = (time.time() + SECONDS_TO_INTERPRET)
        else:
            if time.time() >= self.end_time:
                # Maximum value within dictionary
                d_max = max(self.output_dict, key=self.output_dict.get)
                # Maximum value within dictionary must exceed ACCURACY_TO_INTERPRET and not be 'nothing'
                if self.output_dict[d_max] >= ceil(self.dict_index * ACCURACY_TO_INTERPRET) and d_max != 'nothing':
                    self.output_to_screen(d_max)
                # Reset dict-index and clear the output_dict for next time-frame
                self.dict_index = 0
                self.output_dict.clear()

    @abstractmethod
    def output_to_screen(self, new_addition):
        pass

    @abstractmethod
    def unique_key_functions(self, key):
        pass


class SLType(SavedModel):
    """Sign Language Model Subclass"""
    def __init__(self):
        super().__init__()
        # Load model from model folder
        self.model = tf.keras.models.load_model("model/sign_language_model")
        # Open classes file enclosed in model folder
        f = open("model/sign_language_model/model_classes.txt", 'r')
        # Read class-labels and format
        self.class_labels = (f.read().strip('[').strip(']')).split(', ')
        self.char_per_line = 30
        f.close()

    def output_to_screen(self, new_addition):
        """Rewrite contained output to screen

        Variable(s):
            new_addition: New classification determined by key_value function
        Return: New output string
        """
        # If the new addition is the 'shift' sign, call the shift_key function to check if
        # the previous two letters need to be changed.
        if new_addition == 'shift':
            self.shift_key()
        else:
            self.output += new_addition
        # Underscore is used as a 'space' placeholder for easier understandability by user
        # Replace all underscores with spaces
        self.output = self.output.replace("_", " ")
        # Automatically add a \n every 'char_per_line' characters in output
        if (len(self.output) % self.char_per_line == 0) and (len(self.output) != 0):
            self.output += "\n"

        return self.output

    def unique_key_functions(self, key):
        """Available unique key functions for model"""
        # Create space in output
        if key == ord(' '):
            # self.output += "_"
            self.output += " "

        # Delete last character in output
        if key == ord('\b'):
            self.output = self.output[:-1]


    def shift_key(self):
        """If the most recently read sign was the "shift" sign,
        check the most recent 2 letters in the output.
        "sh" -> "j"
        "s" -> "z"
        Workaround to account for letters not possible in our implementation.
        Directly edits the last two letters of self.output
        """
        # If the last letter of output is 'd' (because it looks similar to z)...
        if self.output[-1:] == 'd':
            # delete them and replace them with 'z'
            self.output = self.output[:-1:] + 'z'
        # If the last letter of output is 'i' (because it looks similar to j)...
        elif self.output[-1:] == 'i':
            # delete them and replace them with 'j'
            self.output = self.output[:-1:] + 'j'


class RPSType(SavedModel):
    """Rock-paper-scissor Model Subclass"""
    def __init__(self):
        super().__init__()
        # Load model from model folder
        self.model = tf.keras.models.load_model("model/rps_model")
        # Open classes file enclosed in model folder
        f = open("model/rps_model/model_classes.txt", 'r')
        # Read class-labels and format
        self.class_labels = (f.read().strip('[').strip(']')).split(', ')
        f.close()

    def output_to_screen(self, new_addition):
        """Rewrite contained output to screen

            Variable(s):
                new_addition: New classification determined by key_value function
            Return: New output string
        """
        # Replace output with new addition
        self.output = new_addition
        return self.output

    def unique_key_functions(self, key):
        """Available unique key functions for model"""
        # Delete last character in output
        if key == ord('\b'):
            self.output = ""
