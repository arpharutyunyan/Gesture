import mediapipe as mp
import cv2
import numpy as np

def orientation(coordinate_landmark_0, coordinate_landmark_9):
    x0 = coordinate_landmark_0.x
    y0 = coordinate_landmark_0.y
    x9 = coordinate_landmark_9.x
    y9 = coordinate_landmark_9.y
    if abs(x9 - x0) < 0.05:
        m = 1000000000
    else:
        m = abs((y9 - y0) / (x9 - x0))
    if m >= 0 and m <= 1:
        if x9 > x0:
            return "Right"
        else:
            return "Left"
    if m > 1:
        if y9 < y0:
            return "Up"
        else:
            return "Down"
        
def dist(l1, l2):
    return ((((l2[0] - l1[0]) ** 2) + ((l2[1] - l1[1]) ** 2)) ** 0.5)

def finger(handlandmarks):
        try:
            needful = [handlandmarks.landmark[0].x, handlandmarks.landmark[0].y]
            d07 = dist(needful, [handlandmarks.landmark[7].x, handlandmarks.landmark[7].y])
            d08 = dist(needful, [handlandmarks.landmark[8].x, handlandmarks.landmark[8].y])
            d011 = dist(needful, [handlandmarks.landmark[11].x, handlandmarks.landmark[11].y])
            d012 = dist(needful, [handlandmarks.landmark[12].x, handlandmarks.landmark[12].y])
            d015 = dist(needful, [handlandmarks.landmark[15].x, handlandmarks.landmark[15].y])
            d016 = dist(needful, [handlandmarks.landmark[16].x, handlandmarks.landmark[16].y])
            d019 = dist(needful, [handlandmarks.landmark[19].x, handlandmarks.landmark[19].y])
            d020 = dist(needful, [handlandmarks.landmark[20].x, handlandmarks.landmark[20].y])
            
            closed = []
            if d07 > d08:
                closed.append(1)
            if d011 > d012:
                closed.append(2)
            if d015 > d016:
                closed.append(3)
            if d019 > d020:
                closed.append(4)
            return closed
        except:
            pass
        
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_hands.Hands() as hands:
        while cap.isOpened():
            
            success, img = cap.read()
            image = img.copy()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    closed = finger(hand_landmarks)
                    if closed == [1, 2, 3, 4]:
                        print("Forward")
                    elif closed == []:
                        print("Back")
                    else:
                        print(orientation(hand_landmarks.landmark[0], hand_landmarks.landmark[9]))
                        
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
                
cap.release()
cv2.destroyAllWindows()