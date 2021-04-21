import cv2
import interpret


def capture(model_to_interpret):

    # Initialize camera
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Interest size, images are saved as capture_zone -10
    capture_zone = 234

    # Width of frame from camera properties
    width = int(capture.get(3))

    if model_to_interpret == 'Sign Language':
        model = interpret.SLType()
    elif model_to_interpret == 'Rock Paper Scissors':
        model = interpret.RPSType()
    else:  # Default to Sign Language interpretation if not sure which to use.
        model = interpret.SLType()

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Defining interest area
        cv2.rectangle(frame, (width - capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)


        # interest = frame[5: capture_zone - 5, width - capture_zone + 5: width - 5]
        # cv2.imwrite('img_to_interpret.png', interest)
        # model.interpret()

        model.unique_key_functions(k)

        cv2.putText(frame, model.output, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    .50, (255, 255, 255), 1)
        # https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
        cv2.imshow("Interpreting", frame)
        k = cv2.waitKey(1)

        # model.keys(k)

        # if k == ord('q'):
            # break


    capture.release()
    cv2.destroyAllWindows()
