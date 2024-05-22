from ultralytics import YOLO
import cv2
import cvzone
import math
import pyttsx3
import pytesseract
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import face_recognition
import os

# creating a list of known persons

path = '../YOLO/knownPersons'
images = []
classNamesKnown = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNamesKnown.append(os.path.splitext(cl)[0])

# mandatory/formality encoding for module to work

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)

# capturing frame in real time

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# setting max numbers of face required to detect(more face leads to slow processing)

detector = FaceMeshDetector(maxFaces=1)

# cap = cv2.VideoCapture("../videos/cars.mp4")

model = YOLO('../yolo-weights/yolov8n.pt')

# coco data set configration class

classNames = [
              "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "hand bag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "surfboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard",
              "cellphone", "microwave", "oven", "toster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "tooth brush"
              ]

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    results = model(img, stream=True)

    # identification of known persons

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    name = ''

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNamesKnown[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    # putting box around detected object

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            # displaying the detected object's text on image

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            # finding the distance of the person
            if faces:
                face = faces[0]
                PointLeft = face[145]
                PointRight = face[374]
                # cv2.line(img, PointLeft, PointRight, (0, 200, 0), 2)
                # cv2.circle(img, PointLeft, 5, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img, PointRight, 5, (255, 0, 255), cv2.FILLED)
                w, _ = detector.findDistance(PointLeft, PointRight)
                W = 6.3
                f = 840
                d = ((W * f) / w)
                # print(d)
                cvzone.putTextRect(img, f'distance: {int(d)}centimeters', (face[10][0] - 125, face[10][1] - 45), scale=2)
            else:
                d = "0"

            # voice feedback of all the stuff detected

            text = f'{classNames[cls]} {conf}, person distance: {int(d)}centimeters', f'{name}'

            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            engine.say(text)
            engine.runAndWait()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
