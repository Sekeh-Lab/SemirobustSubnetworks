"""Contains utility functions for calculating activations and connectivity. Adapted code is acknowledged in comments"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import copy




visualisation = {}

"""
hook_fn(), activations(), and get_all_layers() adapted from: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
"""

#### Hook Function
def hook_fn(m, i, o):
    visualisation[m] = o 


### Create forward hooks to all layers which will collect activation state
def get_all_layers(model, hook_handles, item_key):
    with torch.no_grad():
        for name, module in enumerate(model.named_modules()):
            if name == item_key:
                hook_handles.append(module[1].register_forward_hook(hook_fn))

### Process and record all of the activations for the given pair of layers
def activations(x_input, model, cuda, item_key):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    handles     = []

    ### Set hooks in the appropriate layer dictated by item_key
    get_all_layers(model, handles, item_key)

    with torch.no_grad():
        ### Generate activations for the provided input data
        model(x_input.cuda())

        if temp_op is None:
            temp_op = visualisation[list(visualisation.keys())[0]].cpu().numpy()
        else:
            temp_op = np.vstack((visualisation[list(visualisation.keys())[0]].cpu().numpy(), temp_op))
        
    parents_op = copy.deepcopy(temp_op)
    # Remove all hook handles
    for handle in handles:
        handle.remove()    
    
    del visualisation[list(visualisation.keys())[0]]

    ### Return the activations from the layer dictated by item_key
    return parents_op