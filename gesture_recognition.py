import cv2
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 7
TEST_SIZE = 0.4

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
    
    for dir in range(0, 1):
        # get path for each gesture like "/home/arpine/Desktop/data/0":  
        d = os.path.join(data_dir, f"{str(dir)}")
        # os.listdir(d) return the list of all names of images in that folder
        for image_path in os.listdir(d):
            # get the full path of specific image 
            full_path = os.path.join(data_dir, f"{str(dir)}", image_path)
            # Returns an image that is loaded from the specified file
            image = cv2.imread(full_path)
            # get dimension for each image
            dim = (IMG_WIDTH, IMG_HEIGHT)
            # resized the image
            image_resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            # add image and their directory name to images and labels list
            images.append(image_resized)
            labels.append(dir)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    pass





print(load_data("/home/arpine/Desktop/data/test"))