import cv2
import numpy as np


# define a video capture object
video = cv2.VideoCapture(0)
i = 0
while True:
    # Capture the video frame
    ret, image = video.read()

    # Display the resulting frame
    # to flip the video with 180 degree 
    image = cv2.flip(image, 1)
    
    
    
    cv2.imshow('frame', image)
    cv2.imwrite('/home/arpine/Desktop/data/Frame'+str(i)+'.jpg', image, params=cv2.COLOR_BAYER_BG2GRAY)

    i += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) and 0XFF == ord('q'):
        break



video.release()
cv2.destroyAllWindows()