"""Contains utility functions for calculating activations and connectivity."""

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
from AuxiliaryScripts import mi_estimator


from multiprocessing import Pool


####################################################################################################################
### Activation Functions
####################################################################################################################

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
        found = False
        for name, module in enumerate(model.named_modules()):
            if name == item_key:
                hook_handles.append(module[1].register_forward_hook(hook_fn))
                found = True
                print("Setting hook for layer ", item_key, " module: ", module[0], flush=True)
        if found == False:
            print("Layer ", item_key, " not found!", flush=True)


### This method gets activations for a single layer. It's kept separate to avoid running out of memory from storing all activations simultaneously
def activations_single_layer(x, y, model, cuda, item_key):
    temp_op, temp_label_op = None, None

    attention_layer = False
    if item_key in [8,20,32,44,56,68,80,92, 104,116,128, 140]:
        item_key -= 1
        attention_layer = True

    ### Set hooks in all tunable layers
    handles     = []
    get_single_layer(model, handles, item_key)

    with torch.no_grad():
        ### If we're using a dataloader then we need to iterate over batches, otherwise we need to manually split batches off a larger tensor
        ### Given a set of data from the buffer, we need to still split it by batchsize to ensure it fits in memory
        finalstep = math.ceil(x.size()[0]/256)-1
        for step in range(0,math.ceil(x.size()[0]/256)):
            if step < finalstep:
                x_input = x[step*256:(step+1)*256]
                y_label = y[step*256:(step+1)*256]
            else:
                x_input = x[step*256:]
                y_label = y[step*256:]
            model(x_input.cuda())


            ### For attention layers get only the outputs from the tuple they return
            if attention_layer == True:
                single_acts[list(single_acts.keys())[0]] = single_acts[list(single_acts.keys())[0]][0]
            #!# Detach may not be needed, double check
            if step == 0:
                temp_op        = single_acts[list(single_acts.keys())[0]].detach().cpu()
                temp_label_op  = y_label.detach().cpu()
            else:
                temp_op        = torch.cat((temp_op, single_acts[list(single_acts.keys())[0]].detach().cpu()), dim=0)
                temp_label_op  = torch.cat((temp_label_op, y_label.detach().cpu()), dim=0)
            

    # Remove all hook handles
    for handle in handles:
        handle.remove()    
    

    del single_acts[list(single_acts.keys())[0]]

    ### Return the stored activations and corresponding labels
    return temp_op, temp_label_op






















####################################################################################################################
### Metric Functions
####################################################################################################################




def get_counts_single(acts, labels, dataset):
    print(labels.shape)

    if dataset=="CIFAR100":
        print("Getting counts CIFAR100", flush=True)            
        acts_split   = [acts[np.where(labels[:] == j)] for j in range(100)]                                   
        label_counts = np.bincount(labels[:], minlength=100) / len(labels)                                     
    elif dataset == "CIFAR10":
        print("Getting counts CIFAR10", flush=True)            
        acts_split   = [acts[np.where(labels[:] == j)] for j in range(10)]                                   
        label_counts = np.bincount(labels[:], minlength=10) / len(labels)  
    elif dataset == "TinyImagenet":
        print("Getting counts TinyImagenet", flush=True)
        acts_split   = [acts[np.where(labels[:] == j)] for j in range(200)]                                   
        label_counts = np.bincount(labels[:], minlength=200) / len(labels)  

    return acts_split, label_counts




### Given tensors of activations and labels, get per-class Mutual Information values on the attacked data
def get_mi_estimate(acts_parent, labels_parent, acts_child, labels_child, dataset, mode="flatten"):
    # Use outputs of f^(j-3) to f^(j)
    if dataset == "CIFAR10":
        n = 10
    elif dataset == "CIFAR100":
        n = 100
    elif dataset == "TinyImagenet":
        n = 200
  
    ### For ViT first take the average of each embedded dimension across all patches to get average feature presence across the image
    if len(acts_parent.shape) == 3:
        acts_parent = torch.transpose(acts_parent,1,2)
        if mode == "mean":
            acts_parent = acts_parent.mean(dim=2)
    if len(acts_child.shape) == 3:
        acts_child = torch.transpose(acts_child,1,2)
        if mode == "mean":
            acts_child = acts_child.mean(dim=2)
    

    ### Process the activations on the pair of layers from all classes into split numpy arrays for per-class calculation
    acts_parent_numpy = acts_parent.numpy() 
    labels_parent_numpy = labels_parent.numpy() 
    acts_child_numpy = acts_child.numpy() 
    labels_child_numpy = labels_child.numpy() 

    print("Shape of parent acts: ", acts_parent_numpy.shape)
    print("Shape of child acts: ", acts_child_numpy.shape)
    if len(acts_parent_numpy.shape) > 2:    
        print("Reshaping parent activations")                                                                 
        acts_parent_numpy = np.reshape(acts_parent_numpy, (np.shape(acts_parent_numpy)[0], -1))
    if len(acts_child_numpy.shape) > 2:                                                                     
        print("Reshaping child activations")                                                                 
        acts_child_numpy = np.reshape(acts_child_numpy, (np.shape(acts_child_numpy)[0], -1))

    acts_parent_split, label_counts = get_counts_single(acts_parent_numpy, labels_parent_numpy, dataset)
    acts_child_split, _ = get_counts_single(acts_child_numpy, labels_child_numpy, dataset)

    

    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))  # Default to 1 if not set
    print("Number of SLURM-allocated CPUs: ", num_cpus, flush=True)

    mi_est_list, mi_est = [], 0
    with Pool(num_cpus) as pool:
        mi_est_list = pool.starmap(mi_estimator.EDGE, 
                                [(acts_parent_split[i], acts_child_split[i]) for i in range(n)])
        #!# Modified function defaults to avoid using named arguments
        # mi_est_list = pool.starmap(mi_estimator.EDGE, 
        #                         [(acts_parent_split[i], acts_child_split[i], normalize_epsilon=False, L_ensemble=1, stochastic=True) for i in range(n)])

    for i in range(n):              
        mi_est += label_counts[i] * mi_est_list[i]

    return mi_est






### For comparing MI with sensitivity
def get_sensitivity(adv_acts, norm_acts):
    sensitivity = {}

    print("Shape of Norm acts for sensitivity: ", norm_acts.shape)
    eps = 1e-8
    if len(norm_acts.shape) > 3:
        actsdif = torch.linalg.matrix_norm((norm_acts - adv_acts), ord=1)
        filternorms = torch.linalg.matrix_norm(norm_acts, ord=1)
        normdifs = actsdif/(filternorms + eps)
        # print("Normalized differences shape for filter activations: ", normdifs.size())
        ### Get the mean for all filters, then for all images
        normdifs = torch.mean(normdifs,dim=1)
        sensitivity = torch.mean(normdifs)

    else:
        actsdif = torch.abs((norm_acts - adv_acts))
        neuronnorms = torch.abs(norm_acts)
        normdifs = actsdif/(neuronnorms + eps)
        # print("Normalized differences shape for neuron activations: ", normdifs.size())
        ### Get the mean for all neurons, then for all images
    normdifs = torch.mean(normdifs,dim=1)
    sensitivity = torch.mean(normdifs)

    print("sensitivities: ", sensitivity)

    return sensitivity






### Given Calculate the connectivity between a given pair of layers
def calc_conn(acts_parent, acts_child, labels, dataset, mode="mean"):

    p1_op = copy.deepcopy(acts_parent) 
    c1_op = copy.deepcopy(acts_child)

    ### For ViTs we want to transpose the patches and embedded dim to get same-length sequences of each dimension (feature) across all patches
    if len(p1_op.shape) == 3:
        if mode == "mean":
            p1_op = torch.transpose(p1_op,1,2)
            p1_op = p1_op.mean(dim=2)
        print("Averaging ViT activtions to shape: ", p1_op.shape)
    ### Otherwise for convolutional layers we want to average over feature maps
    elif len(p1_op.shape) > 3:
        p1_op = p1_op.mean(dim=3).mean(dim=2)
        
    if len(c1_op.shape) == 3:
        if mode == "mean":
            c1_op = torch.transpose(c1_op,1,2)
            c1_op = c1_op.mean(dim=2)
    elif len(c1_op.shape) > 3:
        c1_op = c1_op.mean(dim=3).mean(dim=2)


    parent_aves = []
    p1_op = p1_op.numpy()
    c1_op = c1_op.numpy()
    
    
    ### Connectivity is standardized by class mean and stdev
    # print("labels: ", list(np.unique(labels.numpy())))
    for label in list(np.unique(labels.numpy())):
        # print("label: ", label, flush=True)
        parent_mask = np.ones(p1_op.shape,dtype=bool)
        child_mask = np.ones(c1_op.shape,dtype=bool)

        parent_mask[labels != label] = False
        parent_mask[:,np.all(np.abs(p1_op) < 0.0001, axis=0)] = False
        child_mask[labels != label] = False
        child_mask[:,np.all(np.abs(c1_op) < 0.0001, axis=0)] = False
        
        p1_op[parent_mask] -= np.mean(p1_op[parent_mask])
        p1_op[parent_mask] /= np.std(p1_op[parent_mask])

        c1_op[child_mask] -= np.mean(c1_op[child_mask])
        c1_op[child_mask] /= np.std(c1_op[child_mask])


    """
    Code for averaging conns by parent prior by layer
    """


    ### Process the activations on the pair of layers from all classes into split numpy arrays for per-class calculation
    labels_numpy = labels.numpy() 

    acts_parent_split, _ = get_counts_single(p1_op, labels_numpy, dataset)
    acts_child_split, _ = get_counts_single(c1_op, labels_numpy, dataset)


    parent_class_aves = []
    parents_by_class = []
    conn_aves = []

    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))  # Default to 1 if not set
    print("Number of SLURM-allocated CPUs: ", num_cpus, flush=True)


    with Pool(num_cpus) as pool:
        parents_by_class = pool.starmap(connectivity_worker_function, 
                                [(acts_parent_split[cl], acts_child_split[cl], cl, mode) for cl in list(np.unique(labels.numpy()))])
    
    # conn_aves = np.mean(np.asarray(parents_by_class), axis=0)
    conn_ave = np.nanmean(np.asarray(parents_by_class))

    return conn_ave
    



def connectivity_worker_function(p1_class, c1_class, cl, mode="mean"):
    ### Parents is a 2D list of all of the connectivities of parents and children for a single class

    ### Flatten patches for all tokens and then
    if mode == "flatten":
        p1_class = np.reshape(p1_class, (-1, np.shape(p1_class)[2]))
        c1_class = np.reshape(c1_class, (-1, np.shape(c1_class)[2]))

    # print("Connectivity acts shape: ", p1_class.shape, " for class: ", cl, flush=True)
    coefs = np.corrcoef(p1_class, c1_class, rowvar=False).astype(np.float32)
    parents = []
    ### Loop over the cross correlation matrix for the rows corresponding to the parent layer's filters
    for i in range(0, len(p1_class[0])):
        # print("looping i : ", i, " in length of p1 class: ", len(p1_class[0]), " in for coefs of size: ", coefs.shape, "coefs: ", coefs, flush=True)
        parents.append(coefs[i, len(p1_class[0]):])

    parents = np.abs(np.asarray(parents))
    return parents















def sigma_estimation(X, Y):
    """ 
    sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med

def distmat(X):
    """ 
    distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    X.view(1, -1)
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D

def kernelmat(X, sigma):
    """ 
    kernel matrix baker
    """
    m = int(X.size()[0])
    # dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])
    Dxx = distmat(X)

    if sigma:
        variance = 2. * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))

    Kxc = torch.mm(Kx, H)

    return Kxc

def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):

    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1E-5         
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)  
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)  
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy






def calc_HSIC_CKA(acts_parent, acts_child, labels, dataset, normalize=False, sigma=2):

    hsic = {}
    p1_op = copy.deepcopy(acts_parent)
    c1_op = copy.deepcopy(acts_child)

    if len(p1_op.shape) > 2:
        p1_op = p1_op.mean(dim=3).mean(dim=2)
    if len(c1_op.shape) > 2:
        c1_op = c1_op.mean(dim=3).mean(dim=2)


    if normalize:
        for label in list(np.unique(labels.numpy())):
            parent_mask = torch.ones_like(p1_op, dtype=torch.bool)
            child_mask = torch.ones_like(c1_op, dtype=torch.bool)
    
            parent_mask[labels != label] = False
            parent_mask[:, torch.all(torch.abs(p1_op) < 0.0001, dim=0)] = False
            child_mask[labels != label] = False
            child_mask[:, torch.all(torch.abs(c1_op) < 0.0001, dim=0)] = False
    
            p1_op[parent_mask] -= torch.mean(p1_op[parent_mask])
            p1_op[parent_mask] /= torch.std(p1_op[parent_mask])
    
            c1_op[child_mask] -= torch.mean(c1_op[child_mask])
            c1_op[child_mask] /= torch.std(c1_op[child_mask])
    


    acts_parent_split, label_counts = get_counts_single(p1_op, labels.numpy(), dataset)
    acts_child_split, _ = get_counts_single(c1_op, labels.numpy(), dataset)

    hsic_total, cka_total = 0,0 
    for cl in list(torch.unique(labels)):

        # print("For class: ", cl, " shape of acts is: ", acts_parent_split[cl].shape, flush=True)
        # print("For class: ", cl.numpy(), " shape of acts is: ", acts_parent_split[cl.numpy()].shape, flush=True)

        temp_hsic = hsic_normalized_cca(acts_parent_split[cl], acts_child_split[cl], sigma=sigma)    
        hsic_total += label_counts[cl] * temp_hsic
        # print("HSIC class done", flush=True)
        cka_total += temp_hsic / np.sqrt(hsic_normalized_cca(acts_parent_split[cl],acts_parent_split[cl], sigma=sigma)
                                         * hsic_normalized_cca(acts_child_split[cl],acts_child_split[cl], sigma=sigma))

    return hsic_total, cka_total













