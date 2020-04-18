Plant Leaf Detection
Identification of plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques.

Overview
Plant Leaf Identification is a system which is able to classify 32 different species of plants on the basis of their leaves using digital image processing techniques. The images are first preprocessed and then their shape, color and texture based features are extracted from the processed image.

A dataset was created using the extracted features to train and test the model. The model used was Support Vector Machine Classifier and was able to classify with 90.05% accuracy.

Dataset
The dataset used is Flavia leaves dataset which can be downloaded from here

Dependencies
Numpy
Pandas
OpenCV
Matplotlib
Scikit Learn
Mahotas
It is recommended to use Anaconda Python 3.6 distribution and using a Jupyter Notebook

Instructions
Create the following folders in the project root -
Flavia leaves dataset : will contain Flavia dataset
mobile captures : will contain mobile captured leaf images for additional testing purposes
Project structure
single_image_process_file.ipynb : contains exploration of preprocessing and feature extraction techniques by operating on a single image
background_subtract_camera_capture_leaf_file.ipynb : contains exploration of techniques to create a background subtraction function to remove background from mobile camera captured leaf images
classify_leaves_flavia.ipynb : uses extracted features as inputs to the model and classifies them using SVM classifier
preprocess_extract_dataset_flavia.ipynb : contains create_dataset() function which performs image pre-processing and feature extraction on the dataset. The dataset is stored in Flavia_features.csv
Methodology
1. Pre-processing
The following steps were followed for pre-processing the image:

Conversion of RGB to Grayscale image
Smoothing image using Guassian filter
Adaptive image thresholding using Otsu's thresholding method
Closing of holes using Morphological Transformation
Boundary extraction using contours
2. Feature extraction
Variou types of leaf features were extracted from the pre-processed image which are listed as follows:

Shape based features : physiological length,physological width, area, perimeter, aspect ratio, rectangularity, circularity
Color based features : mean and standard deviations of R,G and B channels
Texture based features : contrast, correlation, inverse difference moments, entropy
3. Model building and testing
(a) Support Vector Machine Classifier was used as the model to classify the plant species
(b) Features were then scaled using StandardScaler
(c) Also parameter tuning was done to find the appropriate hyperparameters of the model using
