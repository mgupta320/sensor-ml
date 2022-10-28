# Using Machine Learning For Gas Classification for MEMS Gas Multisensor
## Introduction
   The purpose of using machine learning enhanced analysis of the sensor response is to improve both the ability to classify gases and the amount of time required to make those classifications. With the goal of implementing a classification algorithm on an embedded system such as an FPGA, we need to consider both the time-to-classification and the complexity of the algorithm being implemented. We explore three different techniques: a classic multi-layer perceptron (MLP), a 1-D convolutional neural network (CNN) and a temporal convolutional network (TCN) [1]. We build these different model in Python using the [PyTorch ML Framework](https://pytorch.org/) and the [scikit-learn machine learning module](https://scikit-learn.org/stable/) along with several other useful packages.

## Code 
### Dependencies
```
import numpy 
//used throughout our code for dealing with matrices

import seaborn
import pandas
import matplotlib
import scipy
//used to import, visualize, and export data

import torch
import sklearn
//used for ML model creation, testing, and training
```
### Important Files
* model_main.py : This file contains the functions required for the testing, training, validating, and measuring for the ML models. Models can be trained and tested in the form of a grid search
* data/data_processing.py : This file contains a classs that is used to handle all the data. Data must be supplied in the form of a 3d matlab matrix with dimensions (tests, samples, sensor_data)
* models/point_model.py :  This folder contains each of the PyTorch machine learning models used throughout the project. 

## References
* Lea, C., Vidal, R., Reiter, A., Hager, G.D. (2016). Temporal Convolutional Networks: A Unified Approach to Action Segmentation. In: Hua, G., Jégou, H. (eds) Computer Vision – ECCV 2016 Workshops. ECCV 2016. Lecture Notes in Computer Science(), vol 9915. Springer, Cham. https://doi.org/10.1007/978-3-319-49409-8_7
