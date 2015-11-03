# otto-watty
This repo contains the main functionality from my entry in the Kaggle competition, Otto Group Product Classification Challenge. 

## Functionality
The code links togehter the neural network library Lasagne with the hyperparameter optimization package Hyperopt.
The code in utils provide everything needed to build and train a neural network. The script sqrttrans_pca.py runs hyperopt in order to optimize hyperparameters. 

##Dependencies
* Lasagne
* Numpy 
* Hyperopt
* Sklearn

Note: it is probably note easy to get this running, because the code is from May 2015 and Lasagne is under rapid development. In addition hyperopt needs some special attention as it needs an old version of pymongo. 

Also the data needs to be downloaded from Kaggle.
