
import cv2
import numpy as np
import os
import sys
# import tensorflow as tf
# from PIL import Image, ImageFilter
# from datetime import datetime

# from sklearn.model_selection import train_test_split
# from tensorflow.python.ops.gen_math_ops import mod

add =  "/home/arpine/Desktop/DATA"

addr = os.listdir(add)
# del(addr[0])
# del(addr[0])
# del(addr[0])
# del(addr[1])
# del(addr[16])
# del(addr[17])
# del(addr[18])
# del(addr[19])
# del(addr[21])
# addr.remove("__pycache__")
# addr.remove("gesture_recognition.py")
# addr.remove("DATA")
# addr.remove("webcamera.py")
# addr.remove(".git")
# addr.remove("data")
# addr.remove("image.py")
# addr.remove("README.md")

print("start")
# for dir in addr:
#         # get path for each gesture like "/home/arpine/Desktop/data/0": 
#         # add =  "/home/arpine/Desktop/Gesture"
#         # d = os.path.join(add, f"{str(dir)}")
#         # os.listdir(d) return the list of all names of images in that folder
#         ges_path = os.path.join(add, dir)
#         for gesture_path in os.listdir(ges_path):
#             im_path = os.path.join(ges_path, gesture_path)

#             for image_path in os.listdir(im_path):
#                 full_path = os.path.join(im_path, image_path)
#                 # Returns an image that is loaded from the specified file
#                 format = full_path.split(".")[1]
#                 if format == "png" or format == "jpg":
#                     image = cv2.imread(full_path)
#                     # get dimension for each image
#                     # dim = (IMG_WIDTH, IMG_HEIGHT)
#                     # resized the image
#                     # image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#                     filename = os.path.join("/home/arpine/Desktop/Gesture/DATA", gesture_path, f"image_{str(i)}.{format}")
#                     image = cv2.imwrite(filename, image)
#                     i+=1
# print("finish" + i)

for dir in addr:
    print(dir)
    d = os.path.join(add, f"{str(dir)}")
    #ges_path = os.path.join(add, dir)
    image = os.listdir(d)
    for i in range(0, len(image), 2):
        ges_path = os.path.join(d, image[i])
        os.remove(ges_path)
print("finish")