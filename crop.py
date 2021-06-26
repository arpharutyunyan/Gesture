import os
import cv2
import mediapipe
import uuid
import numpy as np

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            data_type = dirpath.split(sep="/")[-1]
            yield [data_type, os.path.abspath(os.path.join(dirpath, f))]


answers = {"palm": 0, "thumb up": 1, "thumb down": 2, "ok": 3, "fist": 4, "l": 5}
handsModule = mediapipe.solutions.hands

for i in absoluteFilePaths("/home/arpine/Desktop/Gesture/DATA"):
        with handsModule.Hands(static_image_mode=True) as hands:

            image = cv2.imread(i[-1])
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    x = [landmark.x for landmark in hand_landmark.landmark]
                    y = [landmark.y for landmark in hand_landmark.landmark]
                
                    center = np.array([np.mean(x)*image_width, np.mean(y)*image_height]).astype('int32')
                    #cv2.circle(image, tuple(center), 10, (255,0,0), 1) #for checking the center
                    #cv2.rectangle(image, (center[0]-128,center[1]-128), (center[0]+128,center[1]+128), (255,0,0), 1)
                    hand = image[center[1]-128:center[1]+128, center[0]-128:center[0]+128]
                    if hand.shape==(256, 256, 3):
                        cv2.imwrite(f"{i[0]}/" + str(uuid.uuid1()) + ".png", hand)
                      