import cv2
import numpy as np
import os
import sys
from datetime import datetime

IMG_WIDTH = 256
IMG_HEIGHT = 196
NUM_CATEGORIES = 4

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for dir in range(0, NUM_CATEGORIES):
        # get path for each gesture like "/home/arpine/Desktop/data/0":  
        d = os.path.join(data_dir, f"{str(dir)}")
        # os.listdir(d) return the list of all names of images in that folder
        for image_path in os.listdir(d):
            # get the full path of specific image 
            full_path = os.path.join(data_dir, f"{str(dir)}", image_path)
            # Returns an image that is loaded from the specified file
            image = cv2.imread(full_path, 0)
            # cv2.imshow("origin", image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("im", image)
            # get dimension for each image
            # dim = (IMG_WIDTH, IMG_HEIGHT)
            # # resized the image
            # image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
            # add image and their directory name to images and labels list
            images.append(image)
            labels.append(dir)

    return images, labels
# start_time = datetime.now()  
# print("Loading ===========")


# images, labels = load_data("/home/arpine/Desktop/test_data")
# images = np.array(images)
# labels = np.array(labels)
# finish_loading_time = datetime.now()
# print("Images load time: ", finish_loading_time - start_time)


# start_time = datetime.now()  
# print("start saving")
# np.save("images_test.npy", images)
# np.save("labels_test.npy", labels)
# # # np.savetxt("images.csv", images, delimiter=',')
# # np.savetxt("labels.csv", labels, delimiter=',')
# finish_loading_time = datetime.now()
# print("Images load time: ", finish_loading_time - start_time)
# images = np.load("images.npy")
# labels = np.load("labels.npy")
full_path = "/home/arpine/Desktop/Gesture/DATA/0/image_4110.jpg"
image = cv2.imread(full_path)
cv2.imshow("origin", image)
# # images = []
# # images.append(image)
# # print(np.array(images).shape)
image = cv2.imread(full_path,0)
# # images = []
# # images.append(image)
# # print(np.array(images).shape)
# cv2.imshow("origin", image)
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # image = cv2.imread(image)
# # image_resized = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
cv2.imshow("gray", image)
cv2.waitKey(0)

# cv2.imshow("im", image)
# # # video.release()       
# cv2.destroyAllWindows()
