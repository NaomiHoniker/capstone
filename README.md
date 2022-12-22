# Champlain College Hand Symbol Interpreter
Created by Naomi Honiker and Trevor Amell's to complete Champlain College Computer Science and Innovation Capstone project requirement.

Project consists of using computer vision and machine learning techniques to analyze American Sign Language hand signs to interpret them into words in hopes to bridge the gap of communication between the hearing-abled and hearing-disabled.

# Additional Libraries and Frameworks Utilized
**OpenCV2** - Used to access the machine's camera for taking images to create the initial models as well as for use during interpretation sessions to capture frames from the camera as well as write on the images itself.

**Tensorflow** - Used to create machine learning models to compare images captured by OpenCV2 and compare to the one's gather within the model itself.

**PySimpleGUI** - Creates UI for user to access the different integrated models as well as an expansion on the data capture file to input more details in an easier fashion.

**NumPy** - Allows for more complex array uses and calculations

**Pillow** - Adds image processing capabilities and more file format support

# How To Install Via Executable Installer
**Executable Hosted Here: https://drive.google.com/file/d/1uGz3phFM2hXaAPh26OM_whMURyjBkYx7/view?usp=sharing**<br>
Run the .exe installer which will give access to all of the above frameworks/libraries for use within the software.

Once the installer has run, open the "capstone" folder and run the "capture_gui.exe" file to begin the interpreter. This may take a few minutes and depending on your system, the console ouput may throw you some tensorflow error, but rest assureced these should not prevent you from utilizing the software.

# How To Install Via Python Project System<br>
Pull the project, direct your favorite IDE to its contents, and run the capture_gui.py file to begin running. Be sure to have all of the available frameworks/libraries installed within an environment your interpreter has access to as downlaoding this way will NOT install the necessary dependencies.

# How to Use
Once the program is running, a simple GUI will appear asking for a model to use. The default models included are one for Rock, Paper, Scissor, and another for Sign Language Fingerspelling. Pick your desired model to test out and the program will then access your camera and begin the interpretation system.

Ensure you're in a well-lit room with a simple background for best use.

The system makes an interpretation every 2 seconds, and must reach an overall accuracy for a single symbol of 75% to print the given symbol. The dictionary output to the console is used to show which symbols were calculated during that sequence. The "Q" key can be used anytime during the interpretation system to return back to the model selection window.

# The Rock, Paper, Scissor Model
**Unique Key Functions:**<br/>
"\b" (Backspace) can be used to delete the current printed interpretation.

# Sign Language Fingerspelling Model
**Unique Key Functions:**<br/>
"\b" (Backspace) can be used to delete the last input letter.

" "(Spacebar) can be used to create a space within the printed interpretation.

**Symbols:**<br>
Reference: https://s3.amazonaws.com/coursestorm/live/media/915bc8ad95b811e88ea30e92b056efc0 <br/>
The above image symbols were used as a reference when taking images to put into the model.

**Z and J:**<br>
The letters "Z" and "J" are moving symbols, which our interpretor doesn't currently have logic to interpret correctly, especially since they're nearly copies of non-moving letters.

"Z"
To interpret the letter "Z", show the camera the hand symbol for "D" and once it's printed on screen, give a thumbs up to the camera in a front facing perspective (nail of your thumb should be facing directly away from the camera). This should change the "D" to a "Z".

"J"
To interpret the letter "J", show the camera the hand symbol for "I" and once it's printed on screen, give a thumbs up to the camera in a front facing perspective (nail of your thumb should be facing directly away from the camera). This should change the "I" to a "J".
