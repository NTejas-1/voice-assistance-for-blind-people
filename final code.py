import math
from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import os
import pyttsx3
import pytesseract
from PIL import Image
import serial
import time
import face_recognition

os.environ["QT_QPA_PLATFORM"] = "xcb"
engine = pyttsx3.init()
engine.setProperty('rate', 120)

serial_port = serial.Serial(
    port="/dev/serial0",
    baudrate=115200,
    timeout=0.5
)

# Face recognition setup
path = 'knownPersons'
images = []
classNamesKnown = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNamesKnown.append(os.path.splitext(cl)[0])

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)
print(f"Number of known encodings: {len(encodeListKnown)}")

# Frame capture function
def capture_frame():
    libcamera_process = subprocess.Popen(
        [
            'libcamera-vid', '--inline', '--framerate', '30', '--width', '640', '--height', '480',
            '--codec', 'mjpeg', '-o', '-'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    print("Capturing frame and saving to the specified path...")

    mjpeg_stream = b""
    saved_frame = None

    try:
        while True:
            chunk = libcamera_process.stdout.read(4096)
            mjpeg_stream += chunk

            start_idx = mjpeg_stream.find(b'\xff\xd8')
            end_idx = mjpeg_stream.find(b'\xff\xd9')

            if start_idx != -1 and end_idx != -1:
                frame_data = mjpeg_stream[start_idx:end_idx + 2]
                mjpeg_stream = mjpeg_stream[end_idx + 2:]

                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    saved_frame = frame
                    break

    except KeyboardInterrupt:
        print("\nProgram interrupted manually.")
    finally:
        libcamera_process.terminate()

        if saved_frame is not None:
            output_dir = os.path.abspath("./test_image")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "captured_frame.jpg")
            cv2.imwrite(output_path, saved_frame)
            print(f"Frame saved at {output_path}")
        else:
            print("No frame was captured.")

def run_object_detection():
    model = YOLO('../YOLO-Weights/yolov8n.pt')
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

    result = model('test_image/captured_frame.jpg', show=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            text = f'{classNames[cls]} {conf}'
            text_to_voice(text)

def run_face_recognition():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNamesKnown[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                text_to_voice(f"Recognized {name}")

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def read_tfluna_data():
    if serial_port.in_waiting >= 9:
        data = serial_port.read(9)
        if data[0] == 0x59 and data[1] == 0x59:
            distance = data[2] + (data[3] << 8)
            strength = data[4] + (data[5] << 8)
            temperature = data[6] + (data[7] << 8)
            print(distance)
            text_to_voice(f'Object at {distance} centimeters')
    time.sleep(0.5)
    serial_port.reset_input_buffer()
    time.sleep(0.5)

def text_to_voice(text):
    lines = text.split("\n")
    for line in lines:
        if line.strip():
            engine.say(line)
            engine.runAndWait()

opt = input(" press \n 1 for object detection with face recognition and LiDAR \n 2 for text extraction")

capture_frame()

if opt == '1':
    run_object_detection()
    run_face_recognition()
    read_tfluna_data()
    cv2.waitKey(0)

elif opt == '2':
    extract_text(r'test_image/captured_frame.jpg', "eng")

else:
    text_to_voice("Select a valid option")
