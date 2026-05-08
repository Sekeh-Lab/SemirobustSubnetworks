"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import argparse
import pickle
import numpy as np
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchnet as tnt

import torchattacks
# from autoattack.autoattack import AutoAttack

from AuxiliaryScripts import utils
from AuxiliaryScripts.utils import *

import warnings
# To prevent PIL warnings.
warnings.filterwarnings("ignore")


######################################################################################################################################################################
###
###     Main function
###
######################################################################################################################################################################


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, trainloader, testloader, f_b_start, root_save_path, starting_dict):
        print("Initializing manager class", flush=True)

        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.root_save_path = root_save_path
        
        self.train_data_loader = trainloader
        self.test_data_loader = testloader
        self.x_test_adv = None
        self.y_test_adv = None
        # self.preds_test_adv = None


        self.criterion = nn.CrossEntropyLoss()



        self.f_b_start = f_b_start
        # self.starting_dict = starting_dict

        self.frozen_dict = self.get_frozen_dict(starting_dict)


        ### Explicitly set frozen layers requires_grad=False. 
        if self.args.frozen_subnet != "none":
            self.set_no_grad()
        # self.set_train_sub()
        self.model.train()
        
        
    def eval(self, biases=None):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None
        ### Ensure you only ever keep the most recent adv test data for use in calculating MI
        self.x_test_norm = None
        self.y_test_norm = None
        self.x_test_adv = None
        self.y_test_adv = None
        # self.preds_test_adv = None

        print('Performing eval...', flush=True)
        
        for batch, label in self.test_data_loader:
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
        
            if self.args.attacktype == "none":
                # print("Clean eval")
                with torch.no_grad():
                    # print("No attack")
                    output = self.model(batch)
           
                if self.x_test_adv is None:
                    self.x_test_norm = batch.detach().cpu()
                    self.y_test_norm = label.detach().cpu()
                    self.x_test_adv = batch.detach().cpu()
                    self.y_test_adv = label.detach().cpu()
                else:
                    self.x_test_norm = torch.cat((self.x_test_norm, batch.detach().cpu()))
                    self.y_test_norm = torch.cat((self.y_test_norm, label.detach().cpu()))
                    self.x_test_adv = torch.cat((self.x_test_adv, batch.detach().cpu()))
                    self.y_test_adv = torch.cat((self.y_test_adv, label.detach().cpu()))

            else:
                if self.args.attacktype == "fgsm":
                    attack = torchattacks.FGSM(self.model, eps=self.args.eps)
                elif self.args.attacktype == "pgd":
                    attack = torchattacks.PGD(self.model, eps=self.args.eps, alpha=self.args.eps_step, steps=self.args.attack_iterations, random_start=True)
                elif self.args.attacktype == "c&w":
                    attack = torchattacks.CW(self.model, c=1, kappa=0, steps=50, lr=0.01)
                elif self.args.attacktype == "autoattack":
                    if self.args.dataset == "CIFAR100":
                        print("Generating CIFAR100 AutoAttack for Eval")
                        attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=100, seed=None, verbose=False)
                    elif self.args.dataset == "TinyImagenet":
                        print("Generating TinyImagenet AutoAttack for Eval")
                        attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=200, seed=None, verbose=False)
                    else:
                        print("Generating CIFAR10 AutoAttack for Eval")
                        attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=10, seed=None, verbose=False)


                X_adv = attack(batch, label)

                with torch.no_grad():
                    # print("Using Attack")
                    output = self.model(X_adv)
                    
                if self.x_test_adv is None:
                    self.x_test_norm = batch.detach().cpu()
                    self.y_test_norm = label.detach().cpu()                    
                    self.x_test_adv = X_adv.detach().cpu()
                    self.y_test_adv =label.detach().cpu()
                else:
                    self.x_test_norm = torch.cat((self.x_test_norm, batch.detach().cpu()))
                    self.y_test_norm = torch.cat((self.y_test_norm, label.detach().cpu()))
                    self.x_test_adv = torch.cat((self.x_test_adv, X_adv.detach().cpu()))
                    self.y_test_adv = torch.cat((self.y_test_adv, label.detach().cpu()))


            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

            # if self.preds_test_adv is None:
            #     self.preds_test_adv = output.detach().cpu().numpy()
            # else:
            #     self.preds_test_adv = np.concatenate((self.preds_test_adv, output.detach().cpu().numpy()))



        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
                                    
        # self.set_train_sub()
        self.model.train()
        return errors




    def train_epoch(self, epoch_idx, optimizer, scheduler=None):
        """Runs model for each batch."""
        for x_nat, y in self.train_data_loader:
            if self.args.defensetype!="none":
                if self.cuda:
                    x_nat = x_nat.cuda()
                    y = y.cuda()
                x_nat = Variable(x_nat, requires_grad=True)
                y = Variable(y)

                attack = torchattacks.PGD(self.model, eps=self.args.eps, alpha=self.args.eps_step, steps=10, random_start=True)
                x_adv = attack(x_nat, y)
                
                x_adv = x_adv.cuda()
                x_adv = Variable(x_adv, requires_grad=True)

                # Set grads to 0.
                optimizer.zero_grad()
                # self.set_train_sub()
        
                # Do forward-backward.
                output = self.model(x_adv)
            else:        
                if self.cuda:
                    x_nat = x_nat.cuda()
                    y = y.cuda()
                x_nat = Variable(x_nat)
                y = Variable(y)
    
                # Set grads to 0.
                optimizer.zero_grad()
        
                # Do forward-backward.
                output = self.model(x_nat)

            # print("Forward done")
            self.criterion(output, y).backward()
            # Update params.
            optimizer.step()
            self.load_frozen()

    
    def train(self, epochs, optimizer, scheduler=None, save=True, target_accuracy=0, delta=0, trial_number=0):
        """Performs training."""

        best_test_accuracy = 0
        os.makedirs(self.root_save_path, exist_ok=True)    
        checkpoint_path = (self.root_save_path + "checkpoint_trial_" + str(trial_number))
        test_error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()


        ### Explicitly set frozen layers requires_grad=False. 
        if self.args.frozen_subnet != "none":
            self.set_no_grad()
        # self.set_train_sub()
        self.model.train()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))
            print("Learning rate:", optimizer.param_groups[0]['lr'])


            self.train_epoch(epoch_idx, optimizer, scheduler=scheduler)
            print("--------------Epoch Training Done")


            if scheduler is not None:
                scheduler.step()

            test_errors = self.eval()
            test_error_history.append(test_errors)
            test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.

            print('Test Accuracy: %0.2f%%' %
                  (test_accuracy))
            
            if test_accuracy >= best_test_accuracy:
                best_test_accuracy=test_accuracy
                if save == True:
                    self.save_model(checkpoint_path)
            
            if target_accuracy != 0 and best_test_accuracy >= (target_accuracy - delta):
                print('Finished finetuning...')
                print('Best test error/accuracy: %0.2f%%, %0.2f%%' %
                      (100 - best_test_accuracy, best_test_accuracy))
                print('-' * 16)
                return best_test_accuracy, idx
                
                
        print('Finished finetuning...')
        print('Best test error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_test_accuracy, best_test_accuracy))
        print('-' * 16)
        return best_test_accuracy, 0
    
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_frozen_dict(self, starting_dict):

        ### Dictionary to store frozen layers for reloading
        partial_state_dict = OrderedDict()

        ### Determine the last layer in F_a based on f_b_start
        cutoff_layer_name = None
        for num, (name, module) in enumerate(self.model.named_modules()):
          if num == self.f_b_start:
            cutoff_layer_name = name
        print("First layer cutoff name for F_b is: ", cutoff_layer_name, flush=True)

        if self.args.frozen_subnet == "fa":
            
            ### Store items until the first layer of f_b is reached
            for k, v in starting_dict.items():
                if cutoff_layer_name in k:
                    break
                else:
                    if k in self.model.state_dict():
                        partial_state_dict[k] = v


        elif self.args.frozen_subnet == "fb":
            
            subnet_end_reached = False
            for k, v in starting_dict.items():
                if subnet_end_reached == False and cutoff_layer_name in k:
                    subnet_end_reached = True
              
                ### Start storing items as soon as the first layer of f_b is reached
                if subnet_end_reached == True:
                    if k in self.model.state_dict():
                        partial_state_dict[k] = v

        

        return partial_state_dict
    
    
    
    
    
    
    
    
    def save_model(self, path):
        print("Saving model to path: ", path)
        torch.save(self.model.state_dict(), path)


    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
            
        
    def load_unfrozen(self, state_dict):
        statedict_copy = copy.deepcopy(state_dict)
        print("\n\n")
        with torch.no_grad():    
            unfrozen_module_states = OrderedDict()

            ### Store all valid items that are not present in the frozen layers (essentially getting the compliment set of modules)
            for k, v in statedict_copy.items():
                if k not in self.frozen_dict and k in self.model.state_dict():
                    unfrozen_module_states[k] = v
                    # print("Adding layer: ", k, " - ", v, " to the Fb partial dict for loading")
            self.model.load_state_dict(unfrozen_module_states, strict=False)
            print("\nReloading ", len(list(unfrozen_module_states.keys())) , "unfrozen layers")





    ### A bit cluttered, but this is used to ensure that frozen weights don't change due to decay
    def load_frozen(self):

        with torch.no_grad():    
            self.model.load_state_dict(self.frozen_dict, strict=False)



    def set_no_grad(self):
        # print("Set train sub")
        if self.args.frozen_subnet == "fa":
            print("Freezing grad fa")
            for name, module in enumerate(self.model.named_modules()):
                if name < self.f_b_start:
                    module[1].requires_grad=False
                else:
                    module[1].requires_grad=True
        elif self.args.frozen_subnet == "fb":
            for name, module in enumerate(self.model.named_modules()):
                if name >= self.f_b_start and self.f_b_start != 0:
                    module[1].requires_grad=False
                else:
                    module[1].requires_grad=True
                


















































