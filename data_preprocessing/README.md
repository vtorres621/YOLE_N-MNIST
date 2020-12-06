# Data Preprocessing 

## File Description

### create_leaky_black.ipynb
- Creates image from event-base data using leaky_surface algorithm.
- Input data: N-MNIST_csv_lim folder (reduced to 1000 train and 100 test .csv files due to github size limit).
- Output data: N-MNIST_leaky folder.


### create_shifted.ipynb
- Creates 128x128 image using one image of shifted MNIST digits.
- Input data: N-MNIST_leaky folder.
- Output data: N-MNIST_leaky_shifted.


## Instructions to run

### Required libraries
- numpy
- cv2
- os
- csv
- pillow
- matplotlib

### create_leaky_black.ipynb
- Unzip N-MNIST_csv_lim.zip.
- Create N-MNIST_leaky folder.
  - Subfolders: Train, Test.
    - Subfolders: 0,1,2,3,4,5,6,7,8,9.
- Run notebook.


### create_shifted.ipynb
- Unzip N-MNIST_leaky.zip.
- Create N-MNIST_leaky_shifted folder.
  - Subfolders: Train, Test.
    - Subfolders: 0,1,2,3,4,5,6,7,8,9.
- Run notebook.
