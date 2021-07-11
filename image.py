
import cv2
import numpy as np
import os

add =  "/home/arpine/Desktop/Gesture/image/jpg/train"
# add = "/home/arpine/Desktop/Gesture/valid"
addr = os.listdir(add)
# print(addr)

print("start")
i = 397

for dir in addr:
        # get path for each gesture like "/home/arpine/Desktop/data/0": 
        # add =  "/home/arpine/Desktop/alberto"
        # d = os.path.join(add, dir)
        print(dir)
        # os.listdir(d) return the list of all names of images in that folder
        ges_path = os.path.join(add, dir) # name
        print(ges_path)
        for gesture_path in os.listdir(ges_path): 
            print(gesture_path)  # l 
            im_path = os.path.join(ges_path, gesture_path) 
            print(im_path)
            # listdir = []
            # for i in range(90):
            image_path = os.listdir(im_path)
            for j in range(len(image_path)):
            # for image_path in os.listdir(im_path):
                # print(image_path)
            # for j in range(90):
                full_path = os.path.join(im_path, image_path[j])
#                 # Returns an image that is loaded from the specified file
                # format = full_path.split(".")[1]
                format = image_path[j].split(".")[1]
                if format == "png" or format == "jpg":
                    # listdir.append(full_path)
                    # os.remove(full_path)
                    image = cv2.imread(full_path)
                    # get dimension for each image
                    # dim = (IMG_WIDTH, IMG_HEIGHT)
                    # resized the image
                    # image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                    filename = os.path.join("/home/arpine/Desktop/Gesture/train", gesture_path, f"image_{str(i)}.{format}")
                    image = cv2.imwrite(filename, image)
                i = i+1
                if j == 39:
                
                    break

            i = i - 40
                # break
        i = i + 40 
            
print("finish" )

# for dir in addr:
#     print(dir)
#     d = os.path.join(add, f"{str(dir)}")
#     #ges_path = os.path.join(add, dir)
#     image = os.listdir(d)
#     for i in range(0, len(image), 2):
#     # for i in image:
#         # full_path = os.path.join(d, i)
#     #     #                 # Returns an image that is loaded from the specified file
#         # format = full_path.split(".")[1]
#         # if format == "png":
#         ges_path = os.path.join(d, image[i])
#         os.remove(ges_path)
# print("finish")


# full_path = "/home/arpine/Desktop/Gesture/image/carmen/ok/000000014.jpg"
# image = cv2.imread(full_path, 0)
# rest, thresh = cv2.threshold(image, 80, 80, cv2.THRESH_BINARY)
# _, contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.imwrite("frame.png", thresh)




# import cv2
# NUM_CATEGORIES = 6

# GESTURE = {0:"ok", 1:"down", 2:"up", 3:"palm", 4:"fist", 5:"l"}

# def load_data(data_dir):
#     for dir in range(0, NUM_CATEGORIES):
#     # adr = "/home/arpine/Desktop/Gesture/poqr/ok/bcbf8425-d850-11eb-9ec6-0ba2456509ee.png"
#         d = os.path.join(data_dir, f"{str(dir)}")
#         # os.listdir(d) return the list of all names of images in that folder
#         for image_path in os.listdir(d):
#         # get the full path of specific image 
#             full_path = os.path.join(data_dir, f"{str(dir)}", image_path)
#             image = cv2.imread(full_path, 0)
#             rest, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
#             _, contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.imwrite(full_path, thresh)