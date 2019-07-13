# import the necessary packages
import imutils
import cv2

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


camera = cv2.VideoCapture(0)

label = "not_smiling"
index = 1

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()


    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:

        roi = gray[fY:fY + fH, fX:fX + fW]
        # roi = cv2.resize(roi, (28, 28))
        cv2.imwrite(f'data//{label}//{label}_img_file_{index}.jpg', roi)

    cv2.imshow("Face", frameClone)

    index += 1

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
