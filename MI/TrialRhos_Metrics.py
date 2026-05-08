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

from AuxiliaryScripts import cifarmodels, data, utils
from AuxiliaryScripts.manager import Manager
import copy


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

FLAGS.add_argument('--measure', choices=['MI', 'Corr', 'Sensitivity'], help='Which measure to use for calculating Rho Information Flow')
FLAGS.add_argument('--measure_mode', choices=['mean', 'flatten'], help='How to process patches/feature maps prior to measure calculation')

   
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
            
            # ### Replace the head for the new number of classes        
            # if args.dataset=="CIFAR10":
            #     image_size = 32
            # elif args.dataset=="CIFAR100":
            #     image_size = 32
            # elif args.dataset=="TinyImagenet":
            #     image_size = 64
            
            # model.image_size = image_size
            # seq_length = model.image_size // model.patch_size
            # full_seq_length = seq_length + 1

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
            


    untrained_dict = copy.deepcopy(model.state_dict())

    if args.defensetype == "pgd":
        robust_path = './saves/' + args.network + "/pgd/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
        print("Loading robust pretrained model from: ", robust_path)

        state_dict = torch.load(robust_path)
        model.load_state_dict(state_dict)
        

        normal_path = './saves/' + args.network + "/none/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
        pretrained_state_dict = torch.load(normal_path)

    elif args.defensetype == "none":
        normal_path = './saves/' + args.network + "/none/" + args.dataset +"/0.031/" + args.run_id + "/semirobust_0"
        print("Loading robust pretrained model from: ", normal_path)

        state_dict = torch.load(normal_path)
        model.load_state_dict(state_dict)
        

        pretrained_state_dict = untrained_dict

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
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    model.cuda()

    

    
    #########################################################################################
    ###    Adversarially Train
    #########################################################################################

    root_save_path = ('./saves/trialrhos/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + 
                                                "/" + args.run_id + "/" + args.attacktype + "/" + str(args.num_fb_layers) + "/")
    os.makedirs(root_save_path,exist_ok=True)
    args.frozen_subnet = "fa"
    

    ### manager with pretrained normal model
    manager = Manager(args, model, trainloader, testloader, f_b_start, root_save_path, state_dict)

    
    
#########################################################################################
###    Run Experiment
#########################################################################################


        
    # Split x_train_adv into lists of samples with each having the same class
    # Find p(y_n) = (# of y_n in data) / N

    ### Set our accuracy threshold and dummy values for the connectivities and accuracy
    # rho_n, rho_n1, rho_n2, rho_n3 = float('inf'), float('inf'), float('inf'), float('inf')
    # results = np.zeros(13)
    num_epochs = args.num_trials
    measureDict = {}
    for measure in ['MI', 'Sensitivity', 'Corr', 'HSIC', 'CKA']:
        # measureDict[measure] = np.zeros((num_epochs, 12))
        measureDict[measure] = np.zeros((num_epochs+2, args.num_fb_layers))
        # [-1]*12
    result_path = root_save_path + 'measureDict_' + args.attacktype + '_' + str(args.num_fb_layers)
    # sensitivities_path = root_save_path + 'sensitivities_' + args.attacktype + '_' + str(args.num_fb_layers) + ".pt"

    # if measures == "MI":
    measures = [args.measure]

    # measures = ["Corr"]
    # measures = ["MI"]
    # measures = ["Sensitivity"]
    # measures = ["MI", "Corr", "CKA"]
    # measures = ["MI", "Sensitivity"]



    print("Start of fully robust calculation")    

    ### Need to run eval to generate the attacked data which will be used for all measures on the current trial
    _ = manager.eval()

    manager.model.eval()
    
    for n in range(0,args.num_fb_layers):

        if f_b_index <= num_layers - (1+n):

            parent_idx = weight_layers[num_layers - (2+n)]
            child_idx = weight_layers[num_layers - (1+n)]

            acts_parent_adv, labels_parent_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, parent_idx)
            acts_child_adv, labels_child_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, child_idx)
            # acts_parent_norm, labels_parent_norm = utils.activations_single_layer(manager.x_test_norm, manager.y_test_norm, manager.model, args.cuda, parent_idx)
            
            # print("\nParent and child labels are equal: ", torch.equal(labels_parent_adv, labels_child_adv))
            # print("Parent and adv_test_labels are equal: ", torch.equal(labels_parent_adv, manager.y_test_adv))

            ### For each measure, use the stored activations of the pair of layers to get the scalar value for the parent layer
            for measure in measures:
                start_time = time.time()
                print("Starting measure: ", measure, flush=True)
                if measure == "MI":
                    measureDict[measure][num_epochs, n] = utils.get_mi_estimate(acts_parent_adv, labels_parent_adv, acts_child_adv, labels_child_adv, args.dataset, args.measure_mode)
                elif measure == "Corr":
                    measureDict[measure][num_epochs, n] = utils.calc_conn(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, args.measure_mode)
                elif measure == "CKA":
                    result_hsic, result_cka = utils.calc_HSIC_CKA(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, normalize=False, sigma=None)
                    measureDict["HSIC"][num_epochs, n] = result_hsic
                    measureDict["CKA"][num_epochs, n] = result_cka
                elif measure == "Sensitivity":
                    measureDict[measure][num_epochs, n] = utils.get_sensitivity(acts_parent_adv, acts_parent_norm)
                print("Ending Time Taken for measure: ", time.time()-start_time, flush=True)

        else:
            for measure in measures:
                if measure == "CKA":
                    measureDict["HSIC"][num_epochs, n] = float('inf')
                    measureDict["CKA"][num_epochs, n] = float('inf')
                else:
                    measureDict[measure][num_epochs, n] = float('inf')


    print("##############################################################")



    print("Start of semirobust calculation")    

    manager.load_unfrozen(pretrained_state_dict)

    ### Need to run eval to generate the attacked data which will be used for all measures on the current trial
    _ = manager.eval()
    manager.model.eval()
    
    for n in range(0,args.num_fb_layers):
        if f_b_index <= num_layers - (1+n):
            parent_idx = weight_layers[num_layers - (2+n)]
            child_idx = weight_layers[num_layers - (1+n)]

            acts_parent_adv, labels_parent_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, parent_idx)
            acts_child_adv, labels_child_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, child_idx)
            # acts_parent_norm, labels_parent_norm = utils.activations_single_layer(manager.x_test_norm, manager.y_test_norm, manager.model, args.cuda, parent_idx)

            # print("\nParent and child labels are equal: ", torch.equal(labels_parent_adv, labels_child_adv))
            # print("Parent and adv_test_labels are equal: ", torch.equal(labels_parent_adv, manager.y_test_adv))

            ### For each measure, use the stored activations of the pair of layers to get the scalar value for the parent layer
            for measure in measures:
                start_time = time.time()
                print("Starting measure: ", measure, flush=True)
                if measure == "MI":
                    measureDict[measure][num_epochs+1, n] = utils.get_mi_estimate(acts_parent_adv, labels_parent_adv, acts_child_adv, labels_child_adv, args.dataset, args.measure_mode)
                elif measure == "Corr":
                    measureDict[measure][num_epochs+1, n] = utils.calc_conn(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, args.measure_mode)
                elif measure == "CKA":
                    result_hsic, result_cka = utils.calc_HSIC_CKA(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, normalize=False, sigma=None)
                    measureDict["HSIC"][num_epochs+1, n] = result_hsic
                    measureDict["CKA"][num_epochs+1, n] = result_cka
                elif measure == "Sensitivity":
                    measureDict[measure][num_epochs+1, n] = utils.get_sensitivity(acts_parent_adv, acts_parent_norm)
    
                print("Ending Time Taken for measure: ", time.time()-start_time, flush=True)


        else:
            for measure in measures:
                if measure == "CKA":
                    measureDict["HSIC"][num_epochs+1, n] = float('inf')
                    measureDict["CKA"][num_epochs+1, n] = float('inf')
                else:
                    measureDict[measure][num_epochs+1, n] = float('inf')

    print("##############################################################")








    for i in range(num_epochs):                               # Number of trials
        print("Start of trial")
        
        checkpoint_path = (root_save_path + "trialcheckpoint_" + str(i))
        trial_checkpoint_state_dict = torch.load(checkpoint_path)
        model.load_state_dict(trial_checkpoint_state_dict)
        

        ### Need to run eval to generate the attacked data which will be used for all measures on the current trial
        _ = manager.eval()
        manager.model.eval()
        
        for n in range(0, args.num_fb_layers):
            if f_b_index <= num_layers - (1+n):
            
                parent_idx = weight_layers[num_layers - (2+n)]
                child_idx = weight_layers[num_layers - (1+n)]

                acts_parent_adv, labels_parent_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, parent_idx)
                acts_child_adv, labels_child_adv = utils.activations_single_layer(manager.x_test_adv, manager.y_test_adv, manager.model, args.cuda, child_idx)
                # acts_parent_norm, labels_parent_norm = utils.activations_single_layer(manager.x_test_norm, manager.y_test_norm, manager.model, args.cuda, parent_idx)
                
                # print("\nParent and child labels are equal: ", torch.equal(labels_parent_adv, labels_child_adv))
                # print("Parent and adv_test_labels are equal: ", torch.equal(labels_parent_adv, manager.y_test_adv))

                ### For each measure, use the stored activations of the pair of layers to get the scalar value for the parent layer
                for measure in measures:
                    start_time = time.time()
                    print("Starting measure: ", measure, flush=True)
                    if measure == "MI":
                        measureDict[measure][i, n] = utils.get_mi_estimate(acts_parent_adv, labels_parent_adv, acts_child_adv, labels_child_adv, args.dataset, args.measure_mode)
                    elif measure == "Corr":
                        measureDict[measure][i, n] = utils.calc_conn(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, args.measure_mode)
                    elif measure == "CKA":
                        result_hsic, result_cka = utils.calc_HSIC_CKA(acts_parent_adv, acts_child_adv, labels_parent_adv, args.dataset, normalize=False, sigma=None)
                        measureDict["HSIC"][i, n] = result_hsic
                        measureDict["CKA"][i, n] = result_cka
                    elif measure == "Sensitivity":
                        measureDict[measure][i, n] = utils.get_sensitivity(acts_parent_adv, acts_parent_norm)

                    print("Ending Time Taken for measure: ", time.time()-start_time, flush=True)


            else:
                for measure in measures:
                    if measure == "CKA":
                        measureDict["HSIC"][i, n] = float('inf')
                        measureDict["CKA"][i, n] = float('inf')
                    else:
                        measureDict[measure][i, n] = float('inf')

        print("##############################################################")
    #     print(results[:])
    #     print("##############################################################")



    # np.save(result_path, measureDict)

    print("\n")
    print(measureDict['MI'])
    print("\n")
    print(measureDict['Corr'])
    print("\n")
    print(measureDict['HSIC'])
    print("\n")
    print(measureDict['CKA'])
    print("\n")
    print(measureDict['Sensitivity'])

    # print("##############################################################")
    # print(results[:])
    # print("##############################################################")






    
if __name__ == '__main__':
    main()
