import cv2
import interpret

# Initialize camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

trigger_rec = False

# Interest size, images are saved as capture_zone -10
capture_zone = 234

# Width of frame from camera properties
width = int(capture.get(3))

rps_model = interpret.RpsSavedModel()

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # Defining interest area
    cv2.rectangle(frame, (width - capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)

    interest = frame[5: capture_zone - 5, width - capture_zone + 5: width - 5]
    cv2.imwrite('img_to_interpret.png', interest)
    text_prediction = rps_model.interpret()

    # https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    cv2.putText(frame, text_prediction, (15, 15), cv2.FONT_HERSHEY_SIMPLEX,
                .50, (255, 255, 255), 1)

    cv2.imshow("Collecting images", frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
