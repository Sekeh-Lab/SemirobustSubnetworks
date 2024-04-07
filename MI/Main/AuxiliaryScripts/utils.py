"""Contains utility functions for calculating activations and connectivity. Adapted code is acknowledged in comments"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


import time
import copy
import math
import sklearn
import random 

import scipy.spatial     as ss

from math                 import log, sqrt
from scipy                import stats
from sklearn              import manifold
from scipy.special        import *
from sklearn.neighbors    import NearestNeighbors

acts = {}

### Returns a hook function directed to store activations in a given dictionary key "name"
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        acts[name] = output.detach().cpu()
    return hook

### Create forward hooks to all layers which will collect activation state
### Collected from ReLu layers when possible, but not all resnet18 trainable layers have coupled relu layers
def get_all_layers(net, hook_handles):
    for module_idx, module in enumerate(net.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print("module_idx: ", module_idx)
            hook_handles.append(module.register_forward_hook(getActivation(module_idx)))


### Process and record all of the activations for the given pair of layers
def activations(X, Y, model, cuda, mean=False):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []
    # print("Size of X is: ", X.size())
    ### A dictionary for storing the activations
    actsdict = {}
    labels = None
    batchsize = 512
    get_all_layers(model,handles)
    with torch.no_grad():
        batches = math.ceil(X.size(0)/batchsize)
        for batch in range(batches):
            if (batch+1) == batches:
                x_input = X[batch*batchsize:]
                y_label = Y[batch*batchsize:]
            else:
                x_input = X[batch*batchsize:(batch+1)*batchsize]
                y_label = Y[batch*batchsize:(batch+1)*batchsize]
            # print("x input size: ", x_input.size(), flush=True)

            model(x_input.cuda())

            if batch == 0:
                labels = y_label.detach().cpu()
                for key in acts.keys():
                    ### For all conv layers we average over the feature maps, this makes them compatible when comparing with linear layers and reduces memory requirements
                    if len(acts[key].shape) > 2 and mean == True:
                        actsdict[key] = acts[key].mean(dim=3).mean(dim=2)
                    else:
                        actsdict[key] = acts[key]
            else: 
                labels = torch.cat((labels, y_label.detach().cpu()),dim=0)
                for key in acts.keys():
                    if len(acts[key].shape) > 2 and mean == True:
                        actsdict[key] = torch.cat((actsdict[key], acts[key].mean(dim=3).mean(dim=2)), dim=0)
                    else:
                        actsdict[key] = torch.cat((actsdict[key], acts[key]), dim=0)

            
    # Remove all hook handles
    for handle in handles:
        handle.remove()    

    return actsdict, labels



single_acts = {}

def hook_fn(m, i, o):
    single_acts[m] = o 


### Create forward hooks to all layers which will collect activation state
def get_single_layer(model, hook_handles, item_key):
    with torch.no_grad():
        for name, module in enumerate(model.named_modules()):
            if name == item_key:
                hook_handles.append(module[1].register_forward_hook(hook_fn))


### This method gets activations for a single layer. It's kept separate to avoid running out of memory from storing all activations simultaneously
def activations_single_layer(x_input, model, cuda, item_key):
    temp_op       = None
    parents_op  = None

    handles     = []

    get_single_layer(model, handles, item_key)

    with torch.no_grad():
        model(x_input.cuda())

        if temp_op is None:
            temp_op        = single_acts[list(single_acts.keys())[0]].cpu().numpy()
        else:
            print("Temp op else clause triggered")
            temp_op        = np.vstack((single_acts[list(single_acts.keys())[0]].cpu().numpy(), temp_op))
        
    parents_op = copy.deepcopy(temp_op)
    # Remove all hook handles
    for handle in handles:
        handle.remove()    
    

    del single_acts[list(single_acts.keys())[0]]

    return parents_op
