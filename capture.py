import cv2


def single_capture():

    # Initialize camera
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    trigger_rec = False
    counter = image_num = 0

    # Interest size, images are saved as capture_zone -10
    capture_zone = 234

    # Width of frame from camera properties
    width = int(capture.get(3))

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        # Defining interest area
        cv2.rectangle(frame, (width-capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)

        if trigger_rec:
            interest = frame[5: capture_zone-5, width-capture_zone+5: width-5]

            cv2.imwrite("test_image", interest)

        else:
            cv2.imshow("Collecting images", frame)
            k = cv2.waitKey(1)

            if k == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()
