import cv2
from datetime import datetime
import os

global thumbs_up, thumbs_down, nothing, one_finger_up, two_finger_up
#desktop_dir = "C:/Users/Trevo/Desktop/"
desktop_dir = "C:/Users/Richa/Desktop/"


def gather_data(num_samples):

    # Initialize camera
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    trigger_rec = False
    counter = 0

    # Interest size, images are saved as capture_zone -10
    capture_zone = 234

    # Width of frame from camera properties
    width = int(capture.get(3))

    # Time-Date And Directory Creation
    now = datetime.now()
    cur_dir = desktop_dir + now.strftime("%d.%m.%Y %H;%M;%S")
    os.mkdir(cur_dir)
    os.chdir(cur_dir)

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if counter == num_samples:
            trigger_rec = not trigger_rec
            counter = 0

        # Defining interest area
        cv2.rectangle(frame, (width-capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)

        if trigger_rec:
            interest = frame[5: capture_zone-5, width-capture_zone+5: width-5]

            counter += 1

            cv2.imwrite((class_name + str(counter) + ".jpg"), interest)

        else:
            cv2.imshow("Collecting images", frame)
            k = cv2.waitKey(1)

            if k == ord('1'):
                trigger_rec = not trigger_rec
                class_name = 'one_finger_up'

            if k == ord('2'):
                trigger_rec = not trigger_rec
                class_name = 'two_finger_up'

            if k == ord('3'):
                trigger_rec = not trigger_rec
                class_name = 'thumbs_up'

            if k == ord('4'):
                trigger_rec = not trigger_rec
                class_name = 'thumbs_down'

            if k == ord('5'):
                trigger_rec = not trigger_rec
                class_name = 'nothing'

            if k == ord('q'):
                break

            try:
                try:
                    os.chdir(cur_dir)
                    os.mkdir(class_name)
                    os.chdir(cur_dir + "/" + class_name)
                except FileExistsError:
                    os.chdir(cur_dir + "/" + class_name)
            except UnboundLocalError:
                pass

    capture.release()
    cv2.destroyAllWindows()
