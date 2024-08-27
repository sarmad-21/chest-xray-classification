# Chest X-Ray Classification for Pneumonia Detection

## Overview 
This project involves developing a computer vision model to classify chest X-ray images into two categories: Normal and Pneumonia. The model is built using TensorFlow and Keras and is trained to detect pneumonia from chest X-ray images. The application is designed to help in medical diagnostics by identifying the presence of pneumonia in X-ray images.

## Technologies Used
- **Python**
- **TensorFlow**
- **Keras**
- **PIL (Python Imaging Library)**
- **NumPy**
- **Scikit-Learn**
- **OS**

## Deep Learning Model
The model uses a Convolutional Neural Network (CNN) architecture to classify images. Key components include:
- **Convolutional Layer**: Extracts features from the input images.
- **ReLU Activation Function**: Introduces non-linearity to the model.
- **Average Pooling Layer**: Reduces the dimensionality of the feature maps.
- **Flattening Layer**: Converts the 2D feature maps into a 1D vector.
- **Fully Connected Layer**: Connects the flattened output to the output layer.
- **Sigmoid Activation Function**: Produces a probability score for binary classification.

## Data 
The dataset consists of chest X-ray images categorized into "Normal" and "Pneumonia" classes. The data is split into training, validation, and test sets. Images are preprocessed by resizing to 32x32 pixels, converting to grayscale, and normalizing. I obtained this dara from 

## Model Training 
The model is trained using the following parameters:
- Batch Size: 5
- Epochs: 20
- Loss Function: Binary Crossentropy
