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
import torch.nn.functional as F
import pickle
import numpy as np
import time
import warnings
import copy
# To prevent PIL warnings.
warnings.filterwarnings("ignore")
from torchmetrics import Accuracy
from torchvision import models

from AuxiliaryScripts import cifarmodels, data
from AuxiliaryScripts.manager import Manager

###General flags
FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--network', choices=['AlexNet', 'VGG16', 'Resnet18', 'Resnet50', 'WideResnet34'], help='Architectures')
FLAGS.add_argument('--attacktype', choices=['none', 'fgsm', 'pgd', 'c&w', 'autoattack'], help='Type of attack used')
FLAGS.add_argument('--defensetype', choices=['none', 'pgd'], default='none', help='Type of attack used')
FLAGS.add_argument('--num_fb_layers', type=int, default=4, help='Number of layers allocated to subnetwork fb')
FLAGS.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'Imagenette2', 'TinyImagenet'], help='Dataset used for training')
FLAGS.add_argument('--batchsize', type=int, default=512, help='Batch size')
FLAGS.add_argument('--trial_epochs', type=int, default=10, help='Number of training epochs')
FLAGS.add_argument('--num_trials', type=int, default=10, help='Number of trials to run')
FLAGS.add_argument('--eps', type=float, default=0.031, help='Perturbation magnitude, to be divided by 255')
FLAGS.add_argument('--eps_step', type=float, default=0.0039, help='Perturbation step size for each iteration, to be divided by 255')
FLAGS.add_argument('--attack_iterations', type=int, default=10, help='Number of iterations in attack')
FLAGS.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
FLAGS.add_argument('--delta', type=float, default=1e-16, help='K Value for Stopping Condition in Alg 1')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--run_id', type=str, default="r000", help='Run identifier')
FLAGS.add_argument('--load_path', type=str, default="", help='Path to trained model state dict')
FLAGS.add_argument('--frozen_subnet', type=str, default="fa", help='Which subnetwork to freeze during training')

   
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
    # torch.cuda.set_device(0)

    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('-'*100)    
    #########################################################################################
    ###    Prepare Data and Loaders
    #########################################################################################
    datavar = data.Dataset(("../data/" + args.dataset), args.dataset, batch_size=args.batchsize)
    
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
    print("X train shape: ", x_train.size(), " Y train shape: ", y_train.size())



    #########################################################################################
    ###    Prepare The Model
    #########################################################################################

    layerdict = {
        "cifar": {
            "VGG16":[2,5,9,12,16,19,22,26,29,32,36,39,42,48,51,54],
            "Resnet18":[1,7,10,13,16,20,23,26,29,32,36,39,42,45,48,52,55,58,61,64,67],
            "WideResnet34": [1,7,10,11,15,18,22,25,29,32,36,39,45,48,49,53,56,60,63,67,70,74,77,83,
                        86,87,91,94,98,101,105,108,112,115,118]
        }
    }

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


    robust_path = './saves/benchmarks/' + args.network + "/pgd/" + args.dataset +"/" + str(args.eps) + "/" + args.trial_id + "/" + args.run_id + "/semirobust_0"
    print("Loading robust pretrained model from: ", robust_path)

    state_dict = torch.load(robust_path)
    model.load_state_dict(state_dict)

    ### Get the number of layers and indices of weight layers in the network for the given datasetif args.dataset=="CIFAR10" or args.dataset=="CIFAR100":
    ### Note: This also includes TinyImagenet as the network architecture is still compatible for 64x64 images due to average pooling
    num_layers = len(layerdict["cifar"][args.network])
    weight_layers = layerdict["cifar"][args.network]
    
    ### Get index of the first layer in f_b
    f_b_index = num_layers - args.num_fb_layers
    f_b_start = 0
    if f_b_index > 0 and f_b_index < num_layers:
        f_b_start = weight_layers[f_b_index] 
    print(f'Set-Up: {f_b_index}, {f_b_start}, {args.attacktype}, {args.network}, {args.eps}')
        
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    model.cuda()

    

    
    #########################################################################################
    ###    Adversarially Train
    #########################################################################################

    root_save_path = './saves/benchmarks/alg1/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + "/" + args.run_id + "/" + args.attacktype + "/" + str(args.num_fb_layers) + "/"
    os.makedirs(root_save_path,exist_ok=True)
    args.frozen_subnet = "fa"
    

    ### manager with pretrained normal model
    manager = Manager(args, model, x_train, x_adv, y_train, testloader, f_b_start, root_save_path, state_dict)


    baseline_norm_error, baseline_error = manager.eval()  
    baseline_norm_acc = 100 - baseline_norm_error[0]
    baseline_acc = 100 - baseline_error[0]
    print("Acc of fully robust model on Norm (X,Y):", baseline_norm_acc)
    print("Acc of fully robust model on Attacked (X,Y):", baseline_acc)

    normal_path = './saves/benchmarks/' + args.network + "/none/" + args.dataset +"/" + str(args.eps) + "/" + args.trial_id + "/" + args.run_id + "/semirobust_0"
    print("Using normal pretrained model from: ", normal_path)
    pretrained_state_dict = torch.load(normal_path)
    
#########################################################################################
###    Run Experiment
#########################################################################################


        
    delta = args.delta
    num_trials = args.num_trials
    results= np.zeros((num_trials, 16))
    result_path = root_save_path + 'results_' + args.attacktype
    manager.x_nat, manager.x_adv = copy.deepcopy(x_train), copy.deepcopy(x_adv)
    manager.y_train = copy.deepcopy(y_train)
    for i in range(num_trials):                               # Number of trials
        print("Start of trial")
        mi_ests = [-1]*12
        # manager.load_unfrozen(state_dict)
        manager.load_unfrozen(pretrained_state_dict)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

        t = time.time()
        trial_test_acc, break_epoch = manager.train(epochs=args.trial_epochs, optimizer=optimizer, target_accuracy=baseline_acc, delta=delta, eval_all=True)   
        t = time.time() - t
        
        # trial_path = root_save_path + 'trialcheckpoint_' + str(i)
        # manager.save_model(trial_path)
        
        
        print("getting estimates of MI")
    
        manager.get_counts()
        for n in range(0,12):
            mi_ests[n] = manager.get_mi_estimate(weight_layers[num_layers - (2+n)],weight_layers[num_layers - (1+n)])  if f_b_index <= num_layers - (1+n) else float('inf')

        # del model
        results[i, 0]=mi_ests[0]
        results[i, 1]=mi_ests[1]
        results[i, 2]=mi_ests[2]
        results[i, 3]=mi_ests[3]
        results[i, 4]=mi_ests[4]
        results[i, 5]=mi_ests[5]
        results[i, 6]=mi_ests[6]
        results[i, 7]=mi_ests[7]
        results[i, 8]=mi_ests[8]
        results[i, 9]=mi_ests[9]
        results[i, 10]=mi_ests[10]
        results[i, 11]=mi_ests[11]
        results[i, 12]=baseline_acc
        results[i, 13]=trial_test_acc
        results[i, 14]=break_epoch
        results[i, 15]=t
        print("##############################################################")
        print(results[:])
        print("##############################################################")


    np.save(result_path, results)
    print("##############################################################")
    print(results[:])
    print("##############################################################")






    
if __name__ == '__main__':
    main()
