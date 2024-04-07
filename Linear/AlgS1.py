"""  Implements the necessary code to run Algorithm 2 from the SM given a pretrained robust network provided as load_path  """

from __future__ import division, print_function
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
import warnings
# To prevent PIL warnings.
warnings.filterwarnings("ignore")
from torchmetrics import Accuracy
from torchvision import models
import linearcifar10models
import data
from torch.autograd import Variable
from torchsummary import summary
import utils
from manager_lincom import Manager

###General flags
FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--network', choices=['Resnet18'], help='Architectures')
FLAGS.add_argument('--attacktype', choices=['none', 'fgsm', 'pgd', 'autoattack'], help='Type of attack used')
FLAGS.add_argument('--defensetype', choices=['none', 'pgd'], default='none', help='Type of attack used')
FLAGS.add_argument('--num_fb_layers', type=int, default=4, help='Number of layers allocated to subnetwork fb')
FLAGS.add_argument('--dataset', choices=['CIFAR10'], help='Dataset used for training')
FLAGS.add_argument('--batchsize', type=int, default=32, help='Batch size')
FLAGS.add_argument('--trial_epochs', type=int, default=10, help='Number of training epochs')
FLAGS.add_argument('--eps', type=float, default=0.031, help='Perturbation magnitude')
FLAGS.add_argument('--eps_step', type=float, default=0.00784, help='Perturbation step size for each iteration')
FLAGS.add_argument('--attack_iterations', type=int, default=10, help='Number of iterations in attack')
FLAGS.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
FLAGS.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
FLAGS.add_argument('--run_id', type=str, default="r000", help='Run identifier')
FLAGS.add_argument('--load_path', type=str, default="", help='Path to trained model state dict')
FLAGS.add_argument('--frozen_subnet', choices=['fa', 'fb', 'none'], default="fa", help='Which subnetwork to freeze during training')
FLAGS.add_argument('--avg', choices=["False","True"], default="True", help='average lambdas over batches to reduce runtime')

   
######################################################################################################################################################################
###
###     Main function
###
######################################################################################################################################################################



def main():
    args = FLAGS.parse_args()
    torch.cuda.set_device(0)

    print("##############################################################\n\
            Network: ", args.network ,"\n\
            Attack: ", args.attacktype ,"\n\
            Loading Path: ", args.load_path ,"\n\
            Run_id: ", args.run_id ,"\n\
            Dataset: ", args.dataset ,"\n\
            Epsilon: ", args.eps ,"\n\
            Epsilon Step: ", args.eps_step ,"\n\
            Attack Iterations: ", args.attack_iterations ,"\n\
            Epochs: ", args.trial_epochs , "\n\
            Average Lambdas: ", args.avg, "\n\
            num_fb_layers: ", args.num_fb_layers, "\n")
    print("##############################################################")
    #########################################################################################
    ###    Prepare Data and Loaders
    #########################################################################################
    datavar = data.Dataset(args, ("../data/" + args.dataset), args.dataset)
    
    if args.dataset == "CIFAR10":
        trainloader, testloader = datavar.cifar10_dataloaders()

 
    #########################################################################################
    ###    Prepare The Model
    #########################################################################################

    ### Defines the layer numbers of each convolutional or linear layer
    ### For uses in splitting the model between Fa and Fb and freezing/loading weights
    layerdict = {
        "cifar": {
            "Resnet18":[1,7,10,13,16,20,23,29,32,36,39,45,48,52,55,61,64,67]
        }
    }

    ### Initialize and load model weights
    model = linearcifar10models.resnet18(args)
    state_dict = torch.load(args.load_path)
    model.load_state_dict(state_dict)

    ### Get the number of layers and indices of weight layers in the network for the given dataset
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

    root_save_path = './saves/' + args.network + "/" + args.defensetype + "/" + args.dataset +"/" + str(args.eps) + "/" + args.run_id + "/alg2trials/" + args.attacktype + "/" + str(args.num_fb_layers) + "/"
    os.makedirs(root_save_path,exist_ok=True)
    
    ### manager with pretrained normal model
    manager = Manager(args, model, trainloader, testloader, f_b_start, root_save_path, state_dict)

    print("\n\n\n")

#########################################################################################
###    Run Experiment
#########################################################################################
    baseline_acc = 0

    ### Get baseline test accuracy of (f_a^*,f_b*)
    baseline_error = manager.eval(lincom=False)
    baseline_acc = 100 - baseline_error[0]

    ### Calculate and store the outputs from (f_a^*,f_b*)
    manager.evalLincom(lincom=False)
    
    ### Get initial accuracy using randomly initialized values of lambda
    lincom_initial_acc = manager.eval(lincom=True)
    lincom_initial_acc = 100 - lincom_initial_acc[0]
    
    ### Calculate optimal lambdas and get resulting accuracy when predicting output as a linear combination of activations in f_a^*
    lincom_optimal_acc = manager.train_lincom()   

    print("baseline acc:", baseline_acc)
    print("initial lincom acc:", lincom_initial_acc)
    print("optimized lincom acc:", lincom_optimal_acc)
    print("Difference from Baseline Acc: ", baseline_acc - lincom_optimal_acc)

#########################################################################################
###    Save Experiment Results
#########################################################################################
    base_path = ("./results/linear/" + args.dataset + "/" + args.network + "/" + args.attacktype + "/" + str(args.num_fb_layers))
    os.makedirs(base_path, exist_ok=True)    
    
    loss_path = (base_path + "/lincom_loss")
    lambda_path = (base_path + "/lambdas")
    os.makedirs(loss_path, exist_ok=True)   
    os.makedirs(lambda_path, exist_ok=True)   
    np.save((lambda_path + "/opt_lambdas.npy"),manager.model.lincom1.lambdas)
    np.save((loss_path + "/lincom_loss.npy"),manager.loss)
    
    results_path = ("./results/" + args.dataset + "/" + args.network + "/" + args.attacktype + "/" + str(args.num_fb_layers))
    os.makedirs(results_path, exist_ok=True)    
    results = np.asarray([baseline_acc, lincom_initial_acc, lincom_optimal_acc, (baseline_acc-lincom_optimal_acc)])
    if args.avg=="True":
        np.save((results_path + "/results_avg_solution.npy"), results)
    else:
        np.save((results_path + "/results.npy"), results)

if __name__ == '__main__':
    main()
