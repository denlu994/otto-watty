"""
This file contains the training loop and is basically just lasagne's mnist
example code. The training loop consists of three functions:
1. A create_iter_function, which defines the functions for a mini-batch.
2. A train function, which uses the create_iter_function to iterate over 
   epochs.
3. A run_train function, which runs the train function for a specified number 
   of epochs and provides some book keeping. 

In addition there are two type run_train functions.
One for running from the command line which has a lot of monitoring print 
statements and one to run with hyperopt which lacks most of the monitoring.

There are also another train loop code with the suffix anneal, which is 
train code which some extensions such as annealing of learning rates and 
max norm contraints. 
"""

from __future__ import print_function

import itertools
import time

import numpy as np

import theano
import theano.tensor as T
import lasagne

# iter_train function     
def create_iter_functions(dataset, output_layer, batch_size,
                          X_tensor_type=T.matrix,
                          trainable_params=None,
                          momentum=0.9,
                          learning_rate=0.1,
                          ):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
                     loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))
    
    if trainable_params is not None:
        all_params = lasagne.layers.get_all_params(output_layer)
        print(trainable_params)
        all_params = [all_params[i] for i in trainable_params]
        print('Trainable params')
        print(all_params)
    else:
        all_params = lasagne.layers.get_all_params(output_layer)
        print('All params trainable')
        print(all_params)
    
    if learning_rate is not None:
        print("Nesterov")
        print(learning_rate)
        updates = lasagne.updates.nesterov_momentum(
            loss_train, all_params, learning_rate, momentum)
    else:
        print("Adadelta")
        updates = lasagne.updates.adadelta(loss_train, all_params)
    
    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
            },
        )
    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            },
        )
    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            },
        )
    
    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        )



def train(iter_funcs, dataset, batch_size):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
       NOTE: Uneven batches are not handled.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)
        
        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }

def run_training_hyperopt(output_layer, dataset, num_epochs=25,
                          batch_size=499,
                          learning_rate=None,
                          momentum=0.9):
    
    iter_funcs = create_iter_functions(dataset=dataset,
                                       output_layer=output_layer,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       momentum=momentum)
    print("Starting training...")
    now = time.time()
    train_loss = []
    valid_loss = []
    for epoch in train(iter_funcs, dataset,batch_size=batch_size):
        print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
        now = time.time()
        print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
        print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
        print("  validation accuracy:\t\t{:.2f} %%".format(
            epoch['valid_accuracy'] * 100))
        train_loss.append(epoch['train_loss'])
        valid_loss.append(epoch['valid_loss'])
        
        if epoch['number'] >= num_epochs:
            break

    valid_loss = np.array(valid_loss)
    train_loss = np.array(train_loss)

    return valid_loss, train_loss


def run_training(output_layer, dataset, num_epochs=25,
                          batch_size=499, 
                          learning_rate=None,
                          trainable_params=None):

    iter_funcs = create_iter_functions(dataset=dataset,
                                       output_layer=output_layer,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       trainable_params=trainable_params)

    print("Starting training...")
    now = time.time()
    try:
        train_loss = []
        valid_loss = []
        best_valid = 100.0
        for epoch in train(iter_funcs, dataset, batch_size=batch_size):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            train_loss.append(epoch['train_loss'])
            valid_loss.append(epoch['valid_loss'])
            
            if epoch['valid_loss'] < best_valid:
                best_valid = epoch['valid_loss']
                best_epoch = epoch['number']
                best_W = lasagne.layers.helper.get_all_param_values(output_layer)
            
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass
    print(lasagne.layers.helper.get_all_layers(output_layer))

    for l in lasagne.layers.helper.get_all_layers(output_layer):
        print(l.get_output_shape())
 
    print("Best validation loss: ", best_valid)
    print("At epoch ", best_epoch, "of ", num_epochs)
    return output_layer, train_loss, valid_loss, best_W
           
############################ANNEAL#############################################
def create_iter_functions_anneal(dataset, output_layer,
                          training_params, batch_size,
                          X_tensor_type=T.matrix,                         
                          ):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    learning_rate = training_params['learning_rate']
    momentum = training_params['momentum']
    print(learning_rate.eval())
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)
    #updates = lasagne.updates.adadelta(loss_train, all_params)
    def norm_const(layer, updates, max_norm):
        all_weight_params = lasagne.layers.helper.get_all_non_bias_params(layer)
        n=0
        for param, update in updates:
            if param in all_weight_params:
                print(param, update)
                updated_W = lasagne.updates.norm_constraint(update, max_norm)
                updates[n]=(param, updated_W)
            n+=1
        return updates
    updates = norm_const(output_layer, updates, 1.5)
    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )
    new_lr = T.fscalar('new_lr')
    update_lr = theano.function([new_lr], [training_params['learning_rate']],
                               updates={training_params['learning_rate']: new_lr},
                               )
    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        learning_rate_update=update_lr
    )

def train_anneal(iter_funcs, dataset, 
                 decay, batch_size):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
       NOTE: Uneven batches are not handled. 
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)
        
        #def decay(ep):
        #    return 0.09*100.0/np.max([epoch,100])
        
        new_learning_rate = np.cast['float32'](decay(epoch))
        new_learn=iter_funcs['learning_rate_update'](new_learning_rate)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            'new_learn': new_learn,
                }


def run_training_anneal(output_layer,dataset,
                        decay,batch_size, 
                        num_epochs=2,                            
                        start_lr=0.09,
                        momentum=0.9):
   
    t_param=dict(learning_rate=theano.shared(np.cast['float32'](start_lr)),
                 momentum=theano.shared(np.cast['float32'](momentum)),)
    iter_funcs = create_iter_functions_anneal(dataset, output_layer,t_param,
                                              batch_size=batch_size)

    print("Starting training...")
    now = time.time()
    try:
        train_loss = []
        valid_loss = []
        best_valid = 100.0
        for epoch in train_anneal(iter_funcs, dataset,decay,batch_size=batch_size):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            print(np.array(epoch['new_learn'][0]))
            train_loss.append(epoch['train_loss'])
            valid_loss.append(epoch['valid_loss'])
            
            if epoch['valid_loss'] < best_valid:
                best_valid = epoch['valid_loss']
                best_epoch = epoch['number']
                best_W = lasagne.layers.helper.get_all_param_values(output_layer)
            
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass
    print(lasagne.layers.helper.get_all_layers(output_layer))

    for l in lasagne.layers.helper.get_all_layers(output_layer):
        print(l.get_output_shape())
    print("Best validation loss: ", best_valid)
    print("At epoch ", best_epoch, "of ", num_epochs)
    return output_layer, train_loss, valid_loss, best_W

def run_training_anneal_hyperopt(output_layer,dataset,
                        decay,batch_size,
                        num_epochs=2,                             
                        start_lr=0.09,
                        momentum=0.9):
   
    t_param=dict(learning_rate=theano.shared(np.cast['float32'](start_lr)),
                 momentum=theano.shared(np.cast['float32'](momentum)),)
    iter_funcs = create_iter_functions_anneal(dataset, output_layer,t_param,
                                              batch_size)

    print("Starting training...")
    now = time.time()
    try:
        train_loss = []
        valid_loss = []
        best_valid = 100.0
        for epoch in train_anneal(iter_funcs, dataset,decay):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            print(np.array(epoch['new_learn'][0]))
            train_loss.append(epoch['train_loss'])
            valid_loss.append(epoch['valid_loss'])
            
            if epoch['valid_loss'] < best_valid:
                best_valid = epoch['valid_loss']
                best_epoch = epoch['number']
                
            if epoch['number'] >= num_epochs:
                break
            
            if np.isnan(epoch['valid_loss']):
                break

            if epoch['number'] > 50 and best_valid > 0.5:          
                break


    except KeyboardInterrupt:
        pass
    
    return train_loss, valid_loss

        
