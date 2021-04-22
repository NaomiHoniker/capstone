import cv2
import interpret
import numpy as np


def capture(model_to_interpret):
    """Capture sequence creation and loop"""

    # Initialize camera
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Interest size, images are saved as capture_zone -10
    capture_zone = 234

    # Width of frame from camera properties
    width = int(capture.get(3))

    # Choose which model to use according to GUI choice
    if model_to_interpret == 'Sign Language':
        model = interpret.SLType()
    elif model_to_interpret == 'Rock Paper Scissors':
        model = interpret.RPSType()
    else:  # Default to Sign Language interpretation if not sure which to use.
        model = interpret.SLType()

    # Capture Loop
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Defining interest area
        cv2.rectangle(frame, (width - capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)
        interest = frame[5: capture_zone - 5, width - capture_zone + 5: width - 5]

        # Write image to drive to interpret
        cv2.imwrite('img_to_interpret.png', interest)
        model.interpret()

        # Measure if text is too long. If so, split it.

        # Rewrite text to screen
        # cv2.putText(frame, model.output, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
        # .50, (255, 255, 255), 1)
        write_text(frame, model.output, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
                   .50, (255, 255, 255), 1)
        # https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
        cv2.imshow("Interpreting", frame)

        # Get key inputs
        k = cv2.waitKey(1)

        # Model-specific key functions
        model.unique_key_functions(k)

        # Quit back to GUI
        if k == ord('q'):
            break

    # Release capture and return to GUI on break
    capture.release()
    cv2.destroyAllWindows()


def write_text(image, output, org, font_face, font_scale, color, thickness):
    """Writes output to the screen. Calls cv2.putText multiple times if necessary.
    Inspired by Eric Cousineau's cv2_puttext_wrapper.py:
    https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592

            Variable(s):
                image: the frame we are passing the text to
                output: string containing the letters read so far
                org: the location of the text
                fontFace: the font
                fontSize: the font size
                color: the color of the text
                thickness: the thickness of the text
                lineType: the type of line being printed
            """
    line_spacing = 1.5
    # Reshape the location where the text will be written
    org = np.array(org, dtype=float)
    assert org.shape == (2,)

    # Splits the output at '\n', which is automatically added in interpret.py
    for line in output.splitlines():
        # Calculate the width and height of the text
        (width, height), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
        bottom_left = org + [0, height]
        new_org = tuple(bottom_left.astype(int))

        cv2.putText(image, line, new_org, font_face, font_scale, color, thickness)
        org += [0, (height * line_spacing)]
