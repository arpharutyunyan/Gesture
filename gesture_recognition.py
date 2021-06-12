from PIL.Image import Image
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image, ImageFilter

from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_math_ops import mod

EPOCHS = 10
IMG_WIDTH = 20
IMG_HEIGHT = 20
NUM_CATEGORIES = 6
TEST_SIZE = 0.4
GESTURE = {0:"start", 1:"up", 2:"down", 3:"right", 4:"left", 5:"stop"}

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python gesture_recognition.py data_directory [model.h5]")
    
    print("Loading ============================================")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print("====================================================")
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE)
   
    # Get a compiled neural network
    model = get_model()
    
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)
   
   #########################################################
    video = cv2.VideoCapture(0)
    
    while True:
        # Capture the video frame
        ret, image = video.read()

        # Display the resulting frame
        # to flip the video with 180 degree 
        image = cv2.flip(image, 1)
        cv2.imshow('frame', image)
        
        # save image for prediction
        image = cv2.imwrite('Frame'+str(0)+'.jpg', image)
        image = "Frame0.jpg"
      
        dim = (IMG_WIDTH, IMG_HEIGHT)
        
        image = tf.keras.preprocessing.image.load_img(image, target_size=dim)
        # Converts a PIL Image instance to a Numpy array. Return a 3D Numpy array.
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # Convert single image to a batch.
        input_arr = np.array([input_arr])
        input_arr = input_arr.astype('float32')/255
        # Generates output predictions for the input samples. Return Numpy array(s) of predictions.
        predictions = model.predict(input_arr)
        # Return the index_array of the maximum values along an axis.
        pre_class = np.argmax(predictions, axis=-1)
        print(GESTURE[pre_class[0]])
    

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

        if cv2.waitKey(1) and 0XFF == ord('q'):
            break



    video.release()
    cv2.destroyAllWindows()
    
    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


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
    # Create a convolutional neural network
    model = tf.keras.models.Sequential(
        [
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', input_shape=((IMG_WIDTH)/2, (IMG_HEIGHT)/2, 3)
        ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        # Add an output layer with output units for all 6 gestures
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Train neural network
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model





if __name__ == "__main__":
    main()