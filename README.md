# Using Machine Learning For Gas Classification for MEMS Gas Multisensor
## Introduction
   The purpose of using machine learning enhanced analysis of the sensor response is to improve both the ability to classify gases and the amount of time required to make those classifications. With the goal of implementing a classification algorithm on an embedded system such as an FPGA, we need to consider both the time-to-classification and the complexity of the algorithm being implemented. We explore two different techniques: a classic artificial neural network (ANN) and a temproal convolutional network (TCN). We build these different model in Python using the [PyTorch ML Framework](https://pytorch.org/) and the [scikit-learn machine learning module](https://scikit-learn.org/stable/) along with several other useful packages.

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
* models/point_model.py :  This file contains the ANN constructed through PyTorch. It take as its inputs the number of hidden layers, the number of nodes in each hidden layer, and the number of sensor responses (6 in the case of the MEMs multisensor)
* models/tcn_model.py : This file contains the TCN constructed through PyTorch. It takes as its inputs the size of the convolving kernel, the number of time steps being inputted, the nnumber of sensor responses, the number of convolutional layers in the model.

