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
    cv2.rectangle(frame, (width-capture_zone, 0), (width, capture_zone), (0, 250, 150), 2)

    interest = frame[5: capture_zone-5, width-capture_zone+5: width-5]
    cv2.imwrite('img_to_interpret.png', interest)
    rps_model.interpret()

    cv2.imshow("Collecting images", frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
