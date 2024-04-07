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
# To prevent PIL warnings.
warnings.filterwarnings("ignore")

import copy

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
FLAGS.add_argument('--eps', type=float, default=0.031, help='Perturbation magnitude, to be divided by 255')
FLAGS.add_argument('--eps_step', type=float, default=0.0039, help='Perturbation step size for each iteration, to be divided by 255')
FLAGS.add_argument('--attack_iterations', type=int, default=10, help='Number of iterations in attack')
FLAGS.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--pretrained', action='store_true', default=False, help='Pretrained non-adv model')
FLAGS.add_argument('--run_id', type=str, default="r000", help='Run identifier')
FLAGS.add_argument('--load_path', type=str, default="", help='Path to trained model state dict')
FLAGS.add_argument('--frozen_subnet', type=str, default="fa", help='Which subnetwork to freeze during training')
FLAGS.add_argument('--eval_trials', action='store_true', default=False, help='Calculate accuracy and rhos for each trial')
FLAGS.add_argument('--untrained_fb', action='store_true', default=False, help='Reload Fb using untrained weights')


   
######################################################################################################################################################################
###
###     Main function
###
######################################################################################################################################################################



def main():
    args = FLAGS.parse_args()
    torch.cuda.set_device(0)

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

    ### Storing an untrained copy for later use in reloading Fb potentially
    untrained_model = copy.deepcopy(model)

    robust_path = './saves/' + args.network + "/pgd/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
    state_dict = torch.load(robust_path)
    model.load_state_dict(state_dict)

    ### Get the number of layers and indices of weight layers in the network for the given dataset if args.dataset=="CIFAR10" or args.dataset=="CIFAR100":
    num_layers = len(layerdict["cifar"][args.network])
    weight_layers = layerdict["cifar"][args.network]
    
    ### Get index of the first layer in f_b
    f_b_index = num_layers - args.num_fb_layers
    f_b_start = 0
    if f_b_index > 0 and f_b_index < num_layers:
        f_b_start = weight_layers[f_b_index] 
    print(f'Set-Up: {f_b_index}, {f_b_start}, {args.attacktype}, {args.network}, {args.eps}')
        
    model.cuda()

    

    
    #########################################################################################
    ###    Adversarially Train
    #########################################################################################
    root_save_path = './saves/trialrhos/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + "/" + args.run_id + "/" + args.attacktype + "/" + str(args.num_fb_layers) + "/"
    os.makedirs(root_save_path,exist_ok=True)
    args.frozen_subnet = "fa"
    
    ### manager with pretrained normal model
    manager = Manager(args, model, trainloader, testloader, f_b_start, root_save_path, state_dict)

    result_path = root_save_path + 'results_' + args.attacktype
    sensitivities_path = root_save_path + 'sensitivities_' + args.attacktype + ".pt"



    #########################################################################################
    ###    Run Experiment
    #########################################################################################

    delta = 1e-16
    results= np.zeros((12, 15))
    mi_ests = [-1]*12
    


    ### FULLY ADVERSARIALLY TRAINED 

    ### Generates and stores the attacked data for getting the MI estimates
    baseline_norm_error, baseline_error = manager.eval()  
    baseline_norm_acc = 100 - baseline_norm_error[0]
    baseline_acc = 100 - baseline_error[0]
    print("Acc of fully robust model on Norm (X,Y):", baseline_norm_acc)
    print("Acc of fully robust model on Attacked (X,Y):", baseline_acc)

    manager.get_counts()
    manager.get_activations(saveID="baseline")
    for n in range(0,12):
        print("For n: ", n, "parent idx: ", weight_layers[num_layers - (2+n)], " child idx: ", weight_layers[num_layers - (1+n)], flush=True)
        mi_ests[n] = manager.get_mi_estimate(weight_layers[num_layers - (2+n)],weight_layers[num_layers - (1+n)])  if f_b_index <= num_layers - (1+n) else float('inf')        

    results[11, 0]=mi_ests[0]
    results[11, 1]=mi_ests[1]
    results[11, 2]=mi_ests[2]
    results[11, 3]=mi_ests[3]
    results[11, 4]=mi_ests[4]
    results[11, 5]=mi_ests[5]
    results[11, 6]=mi_ests[6]
    results[11, 7]=mi_ests[7]
    results[11, 8]=mi_ests[8]
    results[11, 9]=mi_ests[9]
    results[11, 10]=mi_ests[10]
    results[11, 11]=mi_ests[11]
    results[11, 12]=baseline_acc
    results[11, 13]=0
    results[11, 14]=0



    ### SEMI-ROBUST SPLIT
    ### Reload either the pretrained non-robust model weights or untrained equivalents
    ### This is to see the effect of having the model in a non-robust minima prior to adversarially training the fully robust model
    if args.untrained_fb == True:
        pretrained_state_dict = untrained_model.state_dict()
        print("Loading untrained model weights")
    else:
        normal_path = './saves/' + args.network + "/none/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
        pretrained_state_dict = torch.load(normal_path)

    ### Get accuracy and MIs of (f_a^*,f_b)
    manager.load_unfrozen(pretrained_state_dict)

    subnet_norm_error, subnet_error = manager.eval()  
    subnet_norm_acc = 100 - subnet_norm_error[0]
    subnet_acc = 100 - subnet_error[0]
    print("Acc of semirobust model on Norm (X,Y):", subnet_norm_acc)
    print("Acc of semirobust model on Attacked (X,Y):", subnet_acc)

    manager.get_counts()
    manager.get_activations(saveID="Semirobust")
    for n in range(0,12):
        print("For n: ", n, "parent idx: ", weight_layers[num_layers - (2+n)], " child idx: ", weight_layers[num_layers - (1+n)], flush=True)
        mi_ests[n] = manager.get_mi_estimate(weight_layers[num_layers - (2+n)],weight_layers[num_layers - (1+n)])  if f_b_index <= num_layers - (1+n) else float('inf')        

    results[0, 0]=mi_ests[0]
    results[0, 1]=mi_ests[1]
    results[0, 2]=mi_ests[2]
    results[0, 3]=mi_ests[3]
    results[0, 4]=mi_ests[4]
    results[0, 5]=mi_ests[5]
    results[0, 6]=mi_ests[6]
    results[0, 7]=mi_ests[7]
    results[0, 8]=mi_ests[8]
    results[0, 9]=mi_ests[9]
    results[0, 10]=mi_ests[10]
    results[0, 11]=mi_ests[11]
    results[0, 12]=0
    results[0, 13]=subnet_acc
    results[0, 14]=0


    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    mi_ests = [-1]*12
    best_test_acc = 0
    sensitivities = {}
    for i in range(10):                               # Number of trials
        print("Start of trial")
        mi_ests = [-1]*12
        print("dif: ", baseline_acc-best_test_acc)

        ### This is the code for assessing rho over the course of training. We collect MI after each epoch and stop early if we regain full robustness
        #!# Note: The naming of the variables as trials is a bit confusing and is a holdover from the Algorithm 1 code.
        if baseline_acc-best_test_acc > delta: 
            t = time.time()
            trial_test_acc, break_epoch = manager.train(epochs=1, optimizer=optimizer, target_accuracy=best_test_acc, delta=delta)   
            t = time.time() - t
            print("Trial test acc: ", trial_test_acc)
            print("Best test acc: ", best_test_acc)
            if trial_test_acc >= best_test_acc:
                best_test_acc = trial_test_acc
            
            
            trial_path = root_save_path + 'trialcheckpoint_' + str(i)
            manager.save_model(trial_path)
            
            
            print("getting estimates of MI")
            manager.get_counts()
            sensitivities[i] = manager.get_activations(saveID=str(i))
            for n in range(0,12):
                print("For n: ", n, "parent idx: ", weight_layers[num_layers - (2+n)], " child idx: ", weight_layers[num_layers - (1+n)], flush=True)
                mi_ests[n] = manager.get_mi_estimate(weight_layers[num_layers - (2+n)],weight_layers[num_layers - (1+n)])  if f_b_index <= num_layers - (1+n) else float('inf')
    
            # del model
            results[i+1, 0]=mi_ests[0]
            results[i+1, 1]=mi_ests[1]
            results[i+1, 2]=mi_ests[2]
            results[i+1, 3]=mi_ests[3]
            results[i+1, 4]=mi_ests[4]
            results[i+1, 5]=mi_ests[5]
            results[i+1, 6]=mi_ests[6]
            results[i+1, 7]=mi_ests[7]
            results[i+1, 8]=mi_ests[8]
            results[i+1, 9]=mi_ests[9]
            results[i+1, 10]=mi_ests[10]
            results[i+1, 11]=mi_ests[11]
            results[i+1, 12]=baseline_acc
            results[i+1, 13]=trial_test_acc
            results[i+1, 14]=i
            print("##############################################################")
            print(results[:])
            print("##############################################################")
    
        else:
                        # del model
            results[i+1, 0]=-1
            results[i+1, 1]=-1
            results[i+1, 2]=-1
            results[i+1, 3]=-1
            results[i+1, 4]=-1
            results[i+1, 5]=-1
            results[i+1, 6]=-1
            results[i+1, 7]=-1
            results[i+1, 8]=-1
            results[i+1, 9]=-1
            results[i+1, 10]=-1
            results[i+1, 11]=-1
            results[i+1, 12]=-1
            results[i+1, 13]=-1
            results[i+1, 14]=-1
            print("##############################################################")
            print(results[:])
            print("##############################################################")
    

    np.save(result_path, results)
    torch.save(sensitivities, sensitivities_path)
    print("##############################################################")
    print(results[:])
    print("##############################################################")
    print(sensitivities)






    
if __name__ == '__main__':
    main()
