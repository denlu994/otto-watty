import sys

import theano
import theano.tensor as T
import lasagne
import numpy as np

def add_dense_layer(l_in, num_units, 
                    weight_init, leakiness=0):
    """Adds a dense layer to the layer l_in"""

    if weight_init=='orto':    
        l_dense = lasagne.layers.DenseLayer(
          l_in,
          num_units=num_units,
          nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=leakiness),
          W=lasagne.init.Orthogonal()
        )
    else:
        l_dense = lasagne.layers.DenseLayer(
          l_in,
          num_units=num_units,
          nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=leakiness),
          W=lasagne.init.GlorotUniform()
        )
          
    return l_dense

def add_dropout_layer(l_in, p=0.5):
    l_dropout =  lasagne.layers.DropoutLayer(l_in, p=p)
    return l_dropout    

def build_dense_layers(l_in, layer_dict, weight_init, leakiness):
    """
    Builds a dense layers from a nested dict of layer parameters. 

    The single layer dict is {num_units: 16, more_layers: 0} which would create
    a layer with 16 hidden units. If we want two layers we build a nested dict 
    where a single layer dict is add in the more_layers key-word, such as
    {num_units: 16, more_layers: {num_units: 32, more_layers: 0}}. 

    The reason for the nested dict is to interface well with hyperopt's choice
    module. Where you can have a choice of either adding an other layer or 
    stop.

    The dict is used for easy extension of parameters. For example if we would 
    want to add the a different dropout parameter for each layer. Then we would 
    add it to the dict as {num_units: 16, dropout: 0.3, more_layers: 0} and 
    adjust the code appropriately.

    The build is done recursively thourough the nested dict. 

    """
    

    #Add the layers in a list instead of keeping variables in order to be able
    #to have different choices without the need for lots of if statements.
    #For example the choice of adding dropout or max pooling. 

    layers=[l_in]
    
    if layer_dict == 0:
        return l_in
    else:
        num_units = int(layer_dict['num_units'])
        layers.append(add_dense_layer(layers[-1],
                                     num_units,
                                     weight_init,
                                     leakiness))
        #if layer_dict['dropout']:
        layers.append(add_dropout_layer(layers[-1], p=0.5))
            
        return build_dense_layers(layers[-1], 
                                  layer_dict['more_layers'], 
                                  weight_init,
                                  leakiness)


def build_model(space, output_dim=9, 
            num_features=93, batch_size=499):
    """Builds a neural network from a specification dict called space"""

    l_in = lasagne.layers.InputLayer(shape=(batch_size,
                                            num_features))
    l_in_dropout = add_dropout_layer(l_in, p=space['input_dropout'])
    print space['input_dropout']
    print space['leakiness']
    dense_layers_out = build_dense_layers(l_in_dropout,
                                         layer_dict=space['dense_part'],
                                         weight_init=space['weight_init'],
                                         leakiness=space['leakiness'])
    
    for item in lasagne.layers.helper.get_all_layers(dense_layers_out):
        print(item)
    if space['weight_init']=='orto':
        l_out = lasagne.layers.DenseLayer(
            dense_layers_out,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Orthogonal()
        )
    else:
        l_out = lasagne.layers.DenseLayer(
            dense_layers_out,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform()
        )
    return l_out


