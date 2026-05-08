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
from torchmetrics import Accuracy
from torchvision import models

from AuxiliaryScripts import cifarmodels, data
from AuxiliaryScripts import utils
from AuxiliaryScripts.manager import Manager


from multiprocessing import Pool

###General flags
FLAGS = argparse.ArgumentParser()


FLAGS.add_argument('--network', choices=['AlexNet', 'VGG16', 'Resnet18', 'Resnet50', 'WideResnet34', 
                                        'vit_b_4', 'vit_b_8', 'vit_b_16', 'vit_b_32'], help='Architectures')
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
FLAGS.add_argument('--frozen_subnet', type=str, default="fa", help='Which subnetwork to freeze during training')
FLAGS.add_argument('--imagenet_pretrained', action='store_true', default=False, help='Whether to use the ViT Imagenet weights for finetuning')
FLAGS.add_argument('--single_trial_num', type=int, default=-1, help='Do only this trial if given a value')

   
   
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



        
    #########################################################################################
    ###    Prepare The Model
    #########################################################################################

    layerdict = {
        "cifar": {
            "AlexNet":[2,5,8,10,12,15],
            "VGG16":[2,5,9,12,16,19,22,26,29,32,36,39,42,48,51,54],
            "Resnet18":[1,7,10,13,16,20,23,26,29,32,36,39,42,45,48,52,55,58,61,64,67],
            "Resnet50":[1,7,9,11,15,18,20,22,26,28,30,35,37,39,43,46,48,50,54,56,58,62,64,66,71,73,75,79,82,84,86,90,92,94,98,100,
                        102,106,108,110,114,116,118,123,125,127,131,134,136,138,142,144,146,150],
            "WideResnet34": [1,7,10,11,15,18,22,25,29,32,36,39,45,48,49,53,56,60,63,67,70,74,77,83,
                        86,87,91,94,98,101,105,108,112,115,118],
            "vit_b_4": [1,8,12,15,20,24,27,32,36,39,44,48,51,56,60,63,68,72,75,80,84,87,92,96,99, 104,108,111,116,120,123,128,132, 135,140,144,147],
            "vit_b_8": [1,8,12,15,20,24,27,32,36,39,44,48,51,56,60,63,68,72,75,80,84,87,92,96,99, 104,108,111,116,120,123,128,132, 135,140,144,147],
            "vit_b_16":[1,8,12,15,20,24,27,32,36,39,44,48,51,56,60,63,68,72,75,80,84,87,92,96,99, 104,108,111,116,120,123,128,132, 135,140,144,147],
            "vit_b_32":[1,8,12,15,20,24,27,32,36,39,44,48,51,56,60,63,68,72,75,80,84,87,92,96,99, 104,108,111,116,120,123,128,132, 135,140,144,147]
        }
    }

    if args.network == 'Resnet18':
        if args.dataset=="CIFAR10":
            model = cifarmodels.resnet18(dataset="cifar10")
        elif args.dataset=="CIFAR100":
            model = cifarmodels.resnet18(dataset="cifar100")
        elif args.dataset=="TinyImagenet":
            model = cifarmodels.resnet18(dataset="tinyimagenet")

    elif args.network == 'VGG16':
        if args.dataset=="CIFAR10":
            model = cifarmodels.vgg16_bn(dataset="cifar10")
        elif args.dataset=="CIFAR100":
            model = cifarmodels.vgg16_bn(dataset="cifar100")    
        elif args.dataset=="TinyImagenet":
            model = cifarmodels.vgg16_bn(dataset="tinyimagenet")    

    elif args.network == 'WideResnet34':
        if args.dataset=="CIFAR10":
            model = cifarmodels.WideResNet(num_classes=10)
        elif args.dataset=="CIFAR100":
            model = cifarmodels.WideResNet(num_classes=100)
        elif args.dataset=="TinyImagenet":
            model = cifarmodels.WideResNet(num_classes=200)

    ### Note: The pretrained weights are those provided by pytorch, this just allows them to be loaded offline
    elif args.network in ['vit_b_4', 'vit_b_8', 'vit_b_16', 'vit_b_32']:
        ### Need to figure out how best to handle overwriting the sequence length for the positional encoders if I want to use these
        if args.imagenet_pretrained:
            if args.network == 'vit_b_16':
                model = cifarmodels.vit_b_16()
                model.load_state_dict(torch.load("./saves/Imagenet/vit_b_16_pretrained.pt", weights_only=False))

            elif args.network == 'vit_b_32':
                model = cifarmodels.vit_b_32()
                model.load_state_dict(torch.load("./saves/Imagenet/vit_b_32_pretrained.pt", weights_only=False))
            
        else:
            ### Replace the head for the new number of classes        
            if args.dataset=="CIFAR10":
                image_size = 32
            elif args.dataset=="CIFAR100":
                image_size = 32
            elif args.dataset=="TinyImagenet":
                image_size = 64

            if args.network == 'vit_b_4':
                model = cifarmodels.vit_b_4(image_size=image_size)

            elif args.network == 'vit_b_8':
                model = cifarmodels.vit_b_8(image_size=image_size)

            elif args.network == 'vit_b_16':
                model = cifarmodels.vit_b_16(image_size=image_size)

            elif args.network == 'vit_b_32':
                model = cifarmodels.vit_b_32(image_size=image_size)
            




        ### Replace the head for the new number of classes        
        if args.dataset=="CIFAR10":
            model.num_classes = 10
            model.heads[0] = nn.Linear(model.hidden_dim, 10)
        elif args.dataset=="CIFAR100":
            model.num_classes = 100
            model.heads[0] = nn.Linear(model.hidden_dim, 100)
        elif args.dataset=="TinyImagenet":
            model.num_classes = 200
            model.heads[0] = nn.Linear(model.hidden_dim, 200)
            



    robust_path = './saves/' + args.network + "/pgd/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
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

    model.cuda()

    

    
    #########################################################################################
    ###    Adversarially Train
    #########################################################################################

    root_save_path = './saves/alg1/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + "/" + args.run_id + "/" + args.attacktype + "/" + str(args.num_fb_layers) + "/"
    os.makedirs(root_save_path,exist_ok=True)
    args.frozen_subnet = "fa"
    

    ### manager with pretrained normal model
    manager = Manager(args, model, trainloader, testloader, f_b_start, root_save_path, state_dict)

    
    
#########################################################################################
###    Run Experiment
#########################################################################################


    num_trials = args.num_trials
    measureDict = {}
    for measure in ['MI', 'corr', 'HSIC', 'CKA']:
        measureDict[measure] = np.zeros((num_trials+1, args.num_fb_layers))
        # [-1]*12


    if args.single_trial_num != -1:
        result_path = root_save_path + 'measureDict_' + args.attacktype + '_' + str(args.num_fb_layers)
    else:
        result_path = root_save_path + 'measureDict_' + args.attacktype + '_' + str(args.num_fb_layers) + '_' + str(args.single_trial_num)

    measures = ["MI"]

    print("##############################################################")




    for i in range(num_trials):                               # Number of trials
        if args.single_trial_num == -1 or i == args.single_trial_num:
            print("Start of trial ", i)
            
            ### To avoid disruptions I run the trials without calculating metrics in Alg1_NoMI.py, then load the checkpoints here
            checkpoint_path = (root_save_path + "checkpoint_trial_" + str(i))
            trial_checkpoint_state_dict = torch.load(checkpoint_path)
            model.load_state_dict(trial_checkpoint_state_dict)
            

            ### Need to run eval to generate the attacked data which will be used for all measures on the current trial
            _ = manager.eval()
            manager.model.eval()

            
            for n in range(0,args.num_fb_layers):
                if f_b_index <= num_layers - (1+n):
                
                    parent_idx = weight_layers[num_layers - (2+n)]
                    child_idx = weight_layers[num_layers - (1+n)]

                    acts_parent, labels_parent = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, parent_idx)
                    acts_child, labels_child = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, child_idx)
                    
                    print("Shape of parent and child activations for layer ", n, ": ", acts_parent.shape, " - ", acts_child.shape)
                    print("\nParent and child labels are equal: ", torch.equal(labels_parent, labels_child))
                    print("Parent and adv_test_labels are equal: ", torch.equal(labels_parent, manager.y_test_adv))

                    ### For each measure, use the stored activations of the pair of layers to get the scalar value for the parent layer
                    for measure in measures:
                        start_time = time.time()
                        print("Starting measure: ", measure, flush=True)
                        if measure == "MI":
                            measureDict[measure][i, n] = utils.get_mi_estimate(acts_parent, labels_parent, acts_child, labels_child, args.dataset)
                        elif measure == "corr":
                            measureDict[measure][i, n] = utils.calc_conn(acts_parent, acts_child, labels_parent, args.dataset)
                        elif measure == "CKA":
                            result_hsic, result_cka = utils.calc_HSIC_CKA(acts_parent, acts_child, labels_parent, args.dataset, normalize=False, sigma=None)
                            measureDict["HSIC"][i, n] = result_hsic
                            measureDict["CKA"][i, n] = result_cka
                        print("Ending Time Taken for measure: ", time.time()-start_time, flush=True)

                else:
                    for measure in measures:
                        if measure != "CKA":
                            measureDict[measure][i, n] = float('inf')
                        elif measure == "CKA":
                            measureDict["HSIC"][i, n] = float('inf')
                            measureDict["CKA"][i, n] = float('inf')

            print("##############################################################")



    np.save(result_path, measureDict)


    # print("##############################################################")
    # print(results[:])
    # print("##############################################################")






    
if __name__ == '__main__':
    main()
