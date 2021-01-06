# EEE591_Final_Project
Implementation of YOLE object detection and classification on N-MNIST (neuromorphic/event camera) Dataset

## For data pre-processing steps*
Open data preprocessing folder. *(Not required to run ML code)

## To see results:
Open train.ipynb to see neural network summary, training process and accuracy results.

## To run the program:
### Required libraries
- tensorflow 2.0
- sklearn
- numpy 
- pickle
- cv2
- os 
- csv
### Instructions
- Download all files and folders
- Extract /dataset/images.zip
- Open /libs/config.py and modify desired number of epochs and batchsize (to achieve ~90% accuracy run for at least 20 epochs, batchsize 32)
- Run training process with the following command:
'python3 train.py'

## To make predictions from test data:
### Required libraries
- tensorflow 2.0
- numpy 
- mimetypes
- argparse
- pickle
- cv2
- os 

### Instructions
- Extract test.zip
- Run the following command:
'python predict.py --input results/test_paths.txt'
- This will predict bounding box and class from test images specified in results/test_paths.txt (in this case, the first three images from test folder).
- Check results in results folder.


## References
N-MNIST dataset - [link](https://www.garrickorchard.com/datasets/n-mnist)

Asynchronous Convolutional Networks for Object Detectionin Neuromorphic Cameras - [link](https://arxiv.org/abs/1805.07931v3)

Multi-class object detection and bounding box regression tutorial - [link](https://www.pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)

