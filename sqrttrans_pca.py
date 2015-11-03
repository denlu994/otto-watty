from __future__ import print_function

import sys
import pickle    

import numpy as np
import lasagne
import theano
import theano.tensor as T

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, rand
from hyperopt.mongoexp import MongoTrials

sys.path.append("./utils")
import otto_train as otto_tr
import otto_load_data as otto_dl

def objective(space, num_epochs=100):
    """
    Defines the objective function used by hyperopt. The objective function 
    is the best validation score of a trained neural network. 
    """    
    
    #Some weirdness with the scope of hyperopt. 
    import pickle
    import sys

    import numpy as np
    
    import lasagne
    import theano
    import theano.tensor as T
    
    sys.path.append("./utils")
    import otto_train as otto_tr
    import otto_load_data as otto_dl
    import model_building as mb  
    
    
    print(space)
    
    #Train code
    dataset=otto_dl.load_data(transform='sqrt',
                              whiten=True,
                              test_size=0.05)
        
    output_layer = mb.build_model(space,
                               num_features=dataset['input_dim'],
                               output_dim=dataset['output_dim'])
    
    valid_loss, train_loss = otto_tr.run_training_hyperopt(
                                 output_layer=output_layer,
                                 dataset=dataset,
                                 batch_size=334,
                                 num_epochs=num_epochs,
                                 learning_rate=space['learning_rate']) 

    #Monitoring
    best_valid_loss = valid_loss.min()
    #Haha this looks fun
    best_valid_epoch = np.argmin(valid_loss)+1 #correct of 0-index
    corresponding_train_loss = train_loss[best_valid_epoch-1]
    print("Best valid loss: ")
    print(best_valid_loss)
    print("Corresponding train loss: ")
    print(corresponding_train_loss)    
    print("At epoch: ")
    print(best_valid_epoch)
 

    return {
            'loss': str(best_valid_loss),
            'attachments':
            {'valid_loss': pickle.dumps(valid_loss),
             'train_loss': pickle.dumps(train_loss),
             },
           }
    
def main():
    """
    Performs hyperparameter optimization with hyperopt. It consists of three
    steps. First define a trials object connected to a mongo database where all 
    the results will be stored. Secondly define a stochastic search space from
    which hyperopt will sample hyperparameter configurations. Thirdly define
    the define the objective function and run the minimization function.  
    """

    
    trials = MongoTrials('mongo://localhost:1234/otto-sqrt-pca-95-5/jobs',
            exp_key='15-11-03')
    
    #Search space
    space={
        

    'dense_part':  {'num_units' : hp.quniform('DL1', 512, 2048, 512),
                    'more_layers' : 
                        {'num_units' : hp.quniform('DL2', 512, 2048, 512),
                         'more_layers' : 
                          hp.choice('MD2', [0,
                            {'num_units' : hp.quniform('DL3', 512, 2048, 512),
                             'more_layers' : 0,} #DL3
                                        ])},#DL2
                                                },#DL1                                        


    'leakiness' : hp.choice('leak', [0, 0.01, 0.15] ),


    'weight_init' : hp.choice('weight',['orto','uni']),
    'input_dropout' : hp.quniform('p_in', 0.1, 0.4, 0.1),
    
    'learning_rate': hp.choice('lr',[0.001,0.01,0.025,0.05,0.1]),
    }
    
    #Optimize
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100, 
               trials=trials)

    print(trials.losses())
    print(best)

if __name__ == '__main__':
    main()
