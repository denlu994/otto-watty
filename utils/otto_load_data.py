from __future__ import print_function

import numpy as np

import lasagne
import theano
import theano.tensor as T

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _load_data(transform='sqrt'):
    def class2float(str):
        return float(str[-1])
    
    X=np.loadtxt('/home/dennis/data/kaggle-otto/train.csv',
                 skiprows=1, delimiter=',',
                 converters={94: class2float})    
    
    X_train = X[:,1:-1]
    y_train = X[:,-1]
    
    if transform=='sqrt':
        X_train = np.sqrt(X_train)
    elif transform=='log':
        X_train = np.log(X_train+1)
       
    data = X_train, y_train
    
    return data


def load_test_data(transform='sqrt'):
    """
    Loads test data, note that the test data isn't scaled. This needs to be 
    handled separately.
    """

    X_test=np.loadtxt('/home/dennis/data/kaggle-otto/test.csv',
                 skiprows=1, delimiter=',')
    
    X_test = X_test[:,1:]
    if transform=='sqrt':
        X_test = np.sqrt(X_test)
    elif transform=='log':
        X_test = np.log(X_test+1)
    
    return X_test


def load_data(transform='sqrt',whiten=True,
              test_size=0.25, random_state=0):
    """
    Get data with labels, split into training and validation set. Also preform
    preprocessing. 

    """
    data = _load_data(transform=transform)
    X, y = data
    y = y-1

    sss = StratifiedShuffleSplit(y, 1, test_size=test_size, 
                                 random_state=random_state)
   
    for train_index, valid_index in sss:
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]    
    
    ss = StandardScaler()
    
    X_train = ss.fit_transform(X_train)
    X_valid = ss.transform(X_valid)


    if whiten:
        pca_train = PCA(n_components=93,whiten=True)
        X_train = pca_train.fit_transform(X_train)
        X_valid = pca_train.transform(X_valid)
    #Set these so i wont have to redo train code    
    X_test = X_valid
    y_test = y_valid
    
    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=9,  
)
