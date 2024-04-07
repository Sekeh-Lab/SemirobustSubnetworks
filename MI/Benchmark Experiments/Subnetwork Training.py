"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import time
import copy

import warnings
# To prevent PIL warnings.
warnings.filterwarnings("ignore")
from torchmetrics import Accuracy
from torchvision import models
from AuxiliaryScripts import cifarmodels, data
from AuxiliaryScripts.manager import Manager



###General flags
FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--network', choices=[VGG16', 'Resnet18',  'WideResnet34'], help='Architectures')
FLAGS.add_argument('--attacktype', choices=['none', 'fgsm', 'pgd', 'autoattack'], default="none", help='Type of attack used')
FLAGS.add_argument('--defensetype', choices=['none', 'pgd'], help='Type of defense used')
FLAGS.add_argument('--num_fb_layers', type=int, default=4, help='Number of layers allocated to subnetwork fb')
FLAGS.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'Imagenette2', 'TinyImagenet'], help='Dataset used for training')
FLAGS.add_argument('--batchsize', type=int, default=512, help='Batch size')
FLAGS.add_argument('--gamma', type=float, default=0.1, help='Scheduler lr gamma')
FLAGS.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
FLAGS.add_argument('--eps', type=float, default=0.031, help='Perturbation magnitude, to be divided by 255')
FLAGS.add_argument('--eps_step', type=float, default=0.0039, help='Perturbation step size for each iteration, to be divided by 255')
FLAGS.add_argument('--attack_iterations', type=int, default=10, help='Number of iterations in attack')
FLAGS.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--rand_init', action='store_true', default=True, help='Random initialization of attacks')
FLAGS.add_argument('--run_id', type=str, default="r000", help='Run identifier')
FLAGS.add_argument('--frozen_subnet', type=str, default="none", help='Which subnetwork to freeze during training')
FLAGS.add_argument('--pretrained', action='store_true', default=False, help='Whether to use the normally trained network for pretraining')

FLAGS.add_argument('--epochs_reset', type=int, default=10, help='Number of epochs between resetting pertubations')
FLAGS.add_argument('--loss_type', choices=['mat', 'trades'], help='Type of loss used for adversarial training with ATTA')
FLAGS.add_argument('--beta',  default=6.0, type=float,help='regularization, i.e., 1/lambda in TRADES')

FLAGS.add_argument('--trial_id', type=str, default="trial_000", help='Trial identifier')

######################################################################################################################################################################
###
###     Main function
###
######################################################################################################################################################################



def main():
    args = FLAGS.parse_args()
    print("Torch: ", torch.cuda.is_available())
    print("GPUs: ", torch.cuda.device_count())
    torch.cuda.set_device(0)
        
    print('Arguments =')
    for arg in vars(args):
        print('\t'+ arg +':',getattr(args,arg))
    print('-'*100)    
    #########################################################################################
    ###    Prepare Data and Loaders
    #########################################################################################
    datavar = data.Dataset(("../data/" + args.dataset), args.dataset, args.batchsize)
    
    if args.dataset == "CIFAR10":
        trainloader, testloader = datavar.cifar10_dataloaders()
    elif args.dataset == "CIFAR100":
        trainloader, testloader = datavar.cifar100_dataloaders()
    elif args.dataset == "TinyImagenet":
        trainloader, testloader = datavar.tinyimagenet_dataloaders()
    else:
        print("Invalid dataset arg")
        return 0
 
    x_train, y_train = [],[]
    for batch, labels in trainloader:
        x_train.append(batch)
        y_train.append(labels)

    x_train, y_train = torch.cat(x_train), torch.cat(y_train)
    x_adv = copy.deepcopy(x_train)
    print("Checking sum of X_train: ", torch.sum(x_train), flush=True)
    print("X train shape: ", x_train.size(), " Y train shape: ", y_train.size())
    # print("X train device: ", x_train.get_device(), flush=True)
    # print("X adv device: ", x_adv.get_device(), flush=True)


    #########################################################################################
    ###    Prepare The Model
    #########################################################################################

    layerdict = {
        "cifar": {
            "VGG16":[2,5,9,12,16,19,22,26,29,32,36,39,42,48,51,54],
            "Resnet18":[1,7,10,13,16,20,23,26,29,32,36,39,42,45,48,52,55,58,61,64,67],
            "WideResnet34": [1,7,10,11,15,18,22,25,29,32,36,39,45,48,49,53,56,60,63,67,70,74,77,83,
                        86,87,91,94,98,101,105,108,112,115,118]  
    }}

    if args.dataset=="CIFAR10":
        if args.network == 'Resnet18':
            model = cifarmodels.resnet18(dataset="cifar10")
        elif args.network == 'VGG16':
            model = cifarmodels.vgg16_bn(dataset="cifar10")
        elif args.network == 'WideResnet34':
            model = cifarmodels.WideResNet(num_classes=10)
    elif args.dataset=="CIFAR100":
        if args.network == 'Resnet18':
            model = cifarmodels.resnet18(dataset="cifar100")
        elif args.network == 'VGG16':
            model = cifarmodels.vgg16_bn(dataset="cifar100")    
        elif args.network == 'WideResnet34':
            model = cifarmodels.WideResNet(num_classes=100)
    elif args.dataset=="TinyImagenet":
        if args.network == 'Resnet18':
            model = cifarmodels.resnet18(dataset="tinyimagenet")
        elif args.network == 'VGG16':
            model = cifarmodels.vgg16_bn(dataset="tinyimagenet")    
        elif args.network == 'WideResnet34':
            model = cifarmodels.WideResNet(num_classes=200)


    if args.pretrained == True:
        ### Loads from the save location of the vanilla training
        ### Since eps doesnt affect vanilla training we just assume it was set to the default of 0.031 and fb layer num was 0
        load_path = './saves/benchmarks/' + args.network + "/none/" + args.dataset +"/0.031/" + args.trial_id + "/" + args.run_id + "/semirobust_0"
        print("Using normal pretrained model from: ", load_path)

        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)

    ### Get the number of layers and indices of weight layers in the network for the given dataset
    num_layers = len(layerdict["cifar"][args.network])
    weight_layers =  layerdict["cifar"][args.network]
       
    ### Get index of the first layer in f_b
    f_b_index = num_layers - args.num_fb_layers
    f_b_start = 0
    if f_b_index > 0 and f_b_index < num_layers:
        f_b_start = weight_layers[f_b_index] 
    print(f'Set-Up: {f_b_index}, {f_b_start}, {args.defensetype}, {args.network}, {args.eps}')
        
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=(args.epochs/3), gamma=0.1)

    model.cuda()

    

    
    #########################################################################################
    ###    Adversarially Train
    #########################################################################################

            
    root_save_path = './saves/benchmarks/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + "/" + args.trial_id + "/" + args.run_id + "/"

    normal_path = root_save_path + 'normal_' + str(f_b_start)
    os.makedirs(root_save_path,exist_ok=True)    
    torch.save(model.state_dict(), normal_path)
    starting_dict = torch.load(normal_path)

    ### manager with pretrained normal model
    manager = Manager(args, model, x_train, x_adv, y_train, testloader, f_b_start, root_save_path, starting_dict)

    # Calculate pretrained, non-adversarial accuracy.
    pretrained_norm_err, pretrained_err = manager.eval()  
    pretrained_norm_acc = 100 - pretrained_norm_err[0]
    pretrained_acc = 100 - pretrained_err[0]
    print("Acc of (f_a,f_b) on Norm (X,Y):", pretrained_norm_acc)
    print("Acc of (f_a,f_b) on Attacked (X,Y):", pretrained_acc, "\n")

    trt = time.time()
    best_val_accuracy, _ = manager.train(args.epochs, optimizer, scheduler, save=True)   
    trt = time.time() - trt

    print("Time for training: ", trt)
    

    adv_path = root_save_path + 'semirobust_' + str(f_b_start)
    manager.save_model(adv_path)

    
if __name__ == '__main__':
    main()
