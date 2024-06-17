import pygame
import time
import datetime  # 시간을 확인하기 위해 datetime 모듈 추가
import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import tflite_runtime.interpreter as tflite
import os
import requests

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load('/home/juwon/rbp_dnn/rbp_dnn/fire-truck.WAV')

face_cascade_name = '/home/juwon/rbp_dnn/rbp_dnn/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = '/home/juwon/rbp_dnn/rbp_dnn/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

# Load TFLite model
interpreter = tflite.Interpreter(model_path='/home/juwon/rbp_dnn/rbp_dnn/drowsinessDetection.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

SZ = 24
status = 'Awake'
number_closed = 0
closed_limit = 7
show_frame = None
sign = None
color = None

frame_width = 640
frame_height = 480
frame_resolution = (frame_width, frame_height)
frame_rate = 16

# Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=frame_resolution)

# Allow the camera to warm up
time.sleep(0.1)

alert_count = 0  # 경고음 횟수 카운트
alert_threshold = 3  # 경고음이 울리는 횟수 한계

try:
    # Capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        start_time = time.time()
        image = frame.array
        show_frame = image

        height, width = image.shape[:2]
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)
        for (x, y, w, h) in faces:
            show_frame = cv2.rectangle(show_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceROI = frame_gray[y:y + h, x:x + w]

            eyes = eyes_cascade.detectMultiScale(faceROI)
            results = []

            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.25))
                show_frame = cv2.circle(show_frame, eye_center, radius, (0, 255, 255), 2)
                eye = faceROI[y2:y2 + h2, x2:x2 + w2]
                eye = cv2.resize(eye, (SZ, SZ))
                eye = eye / 255.0
                eye = eye.astype(np.float32)
                eye = eye.reshape(SZ, SZ, -1)
                eye = np.expand_dims(eye, axis=0)

                interpreter.set_tensor(input_details[0]['index'], eye)
                interpreter.invoke()
                result = interpreter.get_tensor(output_details[0]['index'])
                results.append(result[0][0])

            if results:  # 결과가 비어 있지 않은지 확인
                if np.mean(results) == 1:
                    color = (0, 255, 0)
                    status = 'Awake'
                    number_closed = number_closed - 1
                    if number_closed < 0:
                        number_closed = 0
                else:
                    color = (0, 0, 255)
                    status = 'Sleep'
                    number_closed = number_closed + 1

                sign = status + ', Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)
                if number_closed > closed_limit:
                    show_frame = frame_gray
                    pygame.mixer.music.play()
                    alert_count += 1  # 경고음 횟수 증가
                    if alert_count >= alert_threshold:
                        try:
                            current_time = datetime.datetime.now().time()  # 현재 시간 확인
                            message = 'Drowsiness detected!'
                            if current_time.hour >= 23:  # 현재 시간이 오후 11시 이후라면
                                message += ' It is late. We recommend reviewing this tomorrow.'
                            
                            response = requests.post('http://192.168.43.71:5000/alert', json={'message': message})
                            if response.status_code == 200:
                                print("Alert message sent successfully.")
                            else:
                                print(f"Failed to send alert message. Status code: {response.status_code}")
                        except Exception as e:
                            print(f"Error sending alert message: {e}")
                        alert_count = 0  # 카운트 초기화
                        time.sleep(10)  # 10초 대기하여 연속 전송 방지

        cv2.putText(show_frame, sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Frame', show_frame)
        
        rawCapture.truncate(0)  # Clear the buffer for the next frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    camera.close()
    cv2.destroyAllWindows()

