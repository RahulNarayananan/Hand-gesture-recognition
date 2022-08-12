import cv2
import mediapipe as mp
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

classifier=Classifier("C:\\Users\\rahul\\Desktop\\DESKTOP\\Python projects\\handrecog\\keras_model.h5", "C:\\Users\\rahul\\Desktop\\DESKTOP\\Python projects\\handrecog\\labels.txt")
mpHands = mp.solutions.hands
hand = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cam=cv2.VideoCapture(0)

ctr=0
offset=20
labels = ["Thumbs up", "Rock", "victory"]

with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    while True:
        webcam,img=cam.read()
        h, w, c = img.shape

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS, 
                                        mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
            for handlms in results.multi_hand_landmarks:
                for lm in handlms.landmark:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handlms.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(image, (x_min-offset, y_min-offset), (x_max+offset, y_max+offset), (0, 255, 0), 2)
                    
                    data=np.ones((300,300,3),np.uint8)*255

                    imgcrop=image[y_min-offset:y_max+offset,x_min-offset:x_max+offset]

                    cropshape=imgcrop.shape
                    
                    aspectRatio = h / w
 
                    if aspectRatio > 1:
                        k = 300 / h
                        wCal = math.ceil(k * w)
                        wGap = math.ceil((300 - wCal) / 2)
                        prediction, index = classifier.getPrediction(imgcrop, draw=False)
                        print(prediction, index)
            
                    else:
                        k = 300 / w
                        hCal = math.ceil(k * h)
                        hGap = math.ceil((300 - hCal) / 2)
                        prediction, index = classifier.getPrediction(imgcrop, draw=False)
                    cv2.putText(image, labels[index], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255), 2)
        cv2.imshow("Image", image)  
        cv2.waitKey(1)
