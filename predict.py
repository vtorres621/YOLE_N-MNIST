from libs import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import pickle
import cv2
import os


#Load image or images


#Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())

#Determine the input file type, assume single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

#If file is txt, then process multiple images
if "text/plain" == filetype:
	# load the image paths in our testing file
	imagePaths = open(args["input"]).read().strip().split("\n")


#Object detector

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())
image_num = 1

#Loop over the images
for imagePath in imagePaths:
    #Load input image from disk and preprocess it
    image = load_img(imagePath, color_mode="grayscale")
    image = img_to_array(image) /255.0
    image = np.expand_dims(image, axis=0)

    #Predict bounding box and class label
    (bboxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = bboxPreds[0]
    
    #Determine the class label with largest predicted probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    #Load input image (Opencv format) and grab dimensions
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    #Scale the predicted bounding box coordinates based on dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    #Draw the predicted bounding box and class label on the image
    y = startY - 7 if startY - 10 > 10 else startY + 12
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
    cv2.rectangle(image, (startX,startY), (endX, endY), (0, 255, 0), 1)

    #Save image in result directory
    cv2.imwrite("results/pred_"+str(image_num)+".png", image)
    image_num+=1

    #Show the ouput image
    cv2.imshow("Output", image)
    cv2.waitKey(0)