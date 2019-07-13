# import the necessary packages
import base64
import requests
import imutils
import cv2
import os
import sys
import time

esp_host = "http://esp52.dept-crp.sashq-r.openstack.sas.com"
http_port = "60011"
import esppy

# Connect to ESP Server
esp = esppy.ESP(f'{esp_host}:{http_port}')
print(f"INFO: Connected to ESP Server :- {esp}")

proj = esp.get_project('rt_cnn_score')

model_request = proj.windows['w_request']

# Load up ASTORE and set it up for hot loading

pub = model_request.create_publisher(blocksize=1, rate=0, pause=0,
                                     dateformat='%Y%m%dT%H:%M:%S.%f', opcode='insert', format='csv')
strToSend = 'i,n,1,"action","load"\n'
pub.send(strToSend)

strToSend = 'i,n,2,"type","astore"\n'
pub.send(strToSend)

strToSend = 'i,n,3,"reference","/mnt/ext1/race_img_bkp_full/smiles/cas_lenet_v2.astore"\n'
pub.send(strToSend)

strToSend = 'i,n,4,,\n'
pub.send(strToSend)

print("INFO: ASTORE loaded into ESP for Scoring")

# Create publishers on the source window so we can send data when needed
src = proj.windows['w_data']
src_pub = src.create_publisher(blocksize=1, rate=0, pause=0,
                               dateformat='%Y%m%dT%H:%M:%S.%f', opcode='insert', format='csv')

print("INFO: Set Up Complete! publisher ready to push data into source")

# subscribing to the model_score window to get the results back.
model_score = proj.windows['w_score']
model_score.subscribe()
print("INFO: Client subscribed to the scoring window")

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# if a video path was not supplied, grab the reference to the webcam
# if not args.get("video", False):
camera = cv2.VideoCapture(0)

# # otherwise, load the video
# else:
#     camera = cv2.VideoCapture(args["video"])
index = 1
# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,

        roi = gray[fY:fY + fH, fX:fX + fW]

        # write ROI to the file system - test this option first if it works
        # remove line later and push v2
        cv2.imwrite('in_files//img_file.jpg', roi)

        # read the image again, move it to ESP
        f = cv2.imread('in_files//img_file.jpg')

        # Encode the image and ship to ESP
        returnvalue, array_in_buffer = cv2.imencode('.jpg', f)
        encoded_string = base64.b64encode(array_in_buffer)

        strToSend = f"i,n,{index}," + encoded_string.decode() + "\n";

        src_pub.send(strToSend)
        print(f"INFO: Frame {index} published into ESP at {time.ctime()}")
        # toy with this - n/w latency - might not need for a true edge deployment
        time.sleep(.15)
        df = model_score.tail(1)[['I__label_', 'P__label_smiling']]
        # might want to refactor this later - but works for the demo
        label = list(df['I__label_'])[0].strip()

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)
    index += 1

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()


# Finally we need to reload the ESP Server after the process stops
request_url = f'{esp_host}:{http_port}/SASESP/server/state'
payload = {"value": "reloaded"}

r = requests.put(request_url, params=payload)
try:
    r.raise_for_status()
    if r.status_code == 200:
        print("INFO: Remote ESP Server Reloaded for another run!")
except requests.HTTPError as e:
    print("ERROR: Could not reload the remote ESP server - shutdown & reload manually")
    print("Error: " + str(e))
