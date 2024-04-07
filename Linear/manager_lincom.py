"""  Manager class for handling training, evaluation, and lambda calculations  """

from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.autograd import Variable
import torchnet as tnt
from torchsummary import summary
import torchattacks

import pickle
import tensorflow as tf
import numpy as np
import pytorch_lightning
import data
from tqdm import tqdm
from utils import *

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
        ### Copy in any command line arguments from the calling script
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.f_b_start = f_b_start
        self.root_save_path = root_save_path
        self.starting_dict = starting_dict
        
        self.train_data_loader = trainloader
        self.test_data_loader = testloader

        self.baseline_acts = []
        self.lincom_acts = []
        self.loss = []

        self.criterion = nn.CrossEntropyLoss()

        ### Explicitly set frozen layers requires_grad=False. 
        if self.args.frozen_subnet != "none":
            self.set_no_grad()
        self.model.train()
        
                
    def eval(self, biases=None, lincom=False):
        """Performs evaluation."""
        self.model.eval()
        ### Dictates whether the model outputs are generated normally or as a linear combination of the activation values in f_a^*
        self.model.lincom=lincom 
        ### Reset the stored activations every time the model is evaluated on the data to ensure a single copy of each sample is stored
        if lincom==True:
            self.model.lincom1.reset_xs()
            
        error_meter = None
 
        print('Performing eval...', flush=True)
        
        for batch, label in self.test_data_loader:
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
    
            ### If no attack is being done, just predict from the input data
            if self.args.attacktype == "none":
                # print("Clean eval")
                with torch.no_grad():
                    # print("No attack")
                    output = self.model(batch)
            ### If an attack is being done, apply the appropriate attack from TorchAttacks library and then predict
            else:
                ### Temporarily disable the use of the linear combination to avoid interfering with attack generation
                self.model.lincom=False
                if self.args.attacktype == "fgsm":
                    attack = torchattacks.FGSM(self.model, eps=self.args.eps)
                elif self.args.attacktype == "pgd":
                    attack = torchattacks.PGD(self.model, eps=self.args.eps, alpha=self.args.eps_step, steps=self.args.attack_iterations, random_start=True)
                elif self.args.attacktype == "autoattack":
                    attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=10, seed=None, verbose=False)

                X_adv = attack(batch, label)
                ### Once attacks are generated, reapply the linear combination setting if True
                self.model.lincom=lincom

                with torch.no_grad():
                    output = self.model(X_adv)
                   

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)


        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
                                    
        self.model.train()
        return errors



    
    
  

   ### Outputs: Accuracy
    ### Stores: 
    ###        f^n acts as baseline_acts
    ###        f_a* acts as lincom_acts
    ###        loss as frobenius norm of lincom_acts-baseline_acts  
    def evalLincom(self, biases=None, lincom=True):
        """Performs evaluation."""
        self.model.eval()
        self.model.lincom=lincom
        self.output = []
        
        print('Performing eval...')
        batchnum=0
        if lincom==True:
            self.model.lincom1.reset_xs()
        
        for batch, label in self.test_data_loader:
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
    
            if self.args.attacktype == "none":
                with torch.no_grad():
                    output = self.model(batch)
            else:
                self.model.lincom=False

                if self.args.attacktype == "fgsm":
                    attack = torchattacks.FGSM(self.model, eps=self.args.eps)
                elif self.args.attacktype == "pgd":
                    attack = torchattacks.PGD(self.model, eps=self.args.eps, alpha=self.args.eps_step, steps=self.args.attack_iterations, random_start=True)
                elif self.args.attacktype == "autoattack":
                    attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=10, seed=None, verbose=False)

                X_adv = attack(batch, label)
                self.model.lincom=lincom

                with torch.no_grad():
                    output = self.model(X_adv)
                    
            
            temp_acts = output
            temp_acts = temp_acts.detach().cpu().numpy()

            ### If linear combinations arent being used, store the output layer activations of (f_a^*,f_b^*)
            if lincom == False:
                if batchnum == 0:
                    self.baseline_acts = []
                    loss = []
                self.baseline_acts.append(temp_acts)
                loss.append(0)
            ### Otherwise store the activations from layers in (f_a^*)
            else:
                if batchnum == 0:
                    self.lincom_acts = []
                    loss=[]
                self.lincom_acts.append(temp_acts)
               
                ptbaseline = torch.from_numpy(self.baseline_acts[batchnum])
                ptlincom = torch.from_numpy(self.lincom_acts[batchnum])
                
                ### not currently used in this setting, but demonstrates the calculation of the loss term
                loss.append(torch.linalg.matrix_norm(ptbaseline-ptlincom))

            batchnum += 1
        ### Ensure no NaN values in the activations
        for i in range(0,len(self.baseline_acts)):
            if np.isnan(np.sum(np.asarray(self.baseline_acts[i]))):
                print("Nan in batch ", i)
        self.model.train()
        return loss
            
    
    ### Calculate and update the lambdas to an optimal value given the stored activations
    def update_lambdas(self):
        ### If averaging, then calculates lambdas for each batch of data and then averages across the batches.
        if self.args.avg == "True":
          print("Using avg")
          batch_lambdas = []
          for i in range(0,len(self.baseline_acts)):
            xs_batch = self.model.lincom1.xs[i]
            baseline_acts_batch = torch.from_numpy(self.baseline_acts[i]).cuda()
            xs_batch_inv = torch.linalg.pinv(xs_batch)
            
            l_batch = torch.mm(xs_batch_inv.type(torch.float64),baseline_acts_batch.type(torch.float64))
            batch_lambdas.append(l_batch.detach().cpu().numpy())
          l_optimal = np.mean(np.asarray(batch_lambdas),axis=0)
        ### Otherwise calculate lambdas using all of the data simultaneously. More accurate but requires more memory and is intractable for larger datasets
        else:
          print("Taking full dataset inverse", flush=True)
          full_baseline_acts = []
          full_xs = []
          for i in range(0,len(self.baseline_acts)):
            a = self.model.lincom1.xs[i].detach().cpu().numpy()
            b = self.baseline_acts[i]
            if i == 0:
              full_xs = a
              full_baseline_acts = b
            else:
              print("Appending batch: ", i, flush=True)
              full_xs = np.concatenate((full_xs,a), axis=0)
              full_baseline_acts = np.concatenate((full_baseline_acts,b), axis=0)
          print("Done appending!", flush=True)
          full_xs = torch.from_numpy(full_xs).cuda()
          full_xs = torch.linalg.pinv(full_xs)
          full_baseline_acts = torch.from_numpy(full_baseline_acts).cuda()

          print("full_xs size: ", full_xs.size(), flush=True)
          print("full_baseline_acts size: ", full_baseline_acts.size(), flush=True)
          l_optimal = torch.mm(full_xs.type(torch.float64),full_baseline_acts.type(torch.float64))
          l_optimal = l_optimal.detach().cpu().numpy()
        self.model.lincom1.lambdas = l_optimal
    

    ### Calculates the activation values in f_a^* and uses them to update lambdas and evaluate accuracy under the linear setting    
    def train_lincom(self):
        self.model.lincom = True      
        self.model = self.model.cuda()
        
        error_history = []

        
        self.model.eval()
        
        ### Calculate loss based on activations and update lambdas
        loss = self.evalLincom(lincom=True)
        self.update_lambdas()
        self.loss = loss
        ### Evaluate accuracy and check early stopping criteria
        errors = self.eval(lincom=True)
        error_history.append(errors)
        accuracy = 100 - errors[0]  # Top-1 accuracy.
        

        self.model.train()

        print('Finished finetuning...')
        print('Accuracy: %0.2f%%, %0.2f%%' %(100 - accuracy, accuracy))
        print('-' * 16)
        return accuracy
    
    
    
    
    
    
    
    
    
    ### Save and load model state dict
    def save_model(self, path):
        print("Saving model to path: ", path)
        torch.save(self.model.state_dict(), path)
       
    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
            
    ### Load the unfrozen subnetwork, either f_a or f_b
    def load_unfrozen(self, state_dict):
        with torch.no_grad():
            if self.args.frozen_subnet == "fa":
                for name, module in enumerate(self.model.named_modules()):
                    if name >= self.f_b_start and self.f_b_start != 0:
                        if isinstance(module[1], nn.BatchNorm2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
                            module[1].running_mean.copy_(state_dict[(module[0] + ".running_mean")])
                            module[1].running_var.copy_(state_dict[(module[0] + ".running_var")])
                            module[1].num_batches_tracked.copy_(state_dict[(module[0] + ".num_batches_tracked")])
                        elif isinstance(module[1], nn.Conv2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                        elif isinstance(module[1], nn.Linear):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
            if self.args.frozen_subnet == "fb":
                print("Loading unfrozen fb")
                for name, module in enumerate(self.model.named_modules()):
                    if name < self.f_b_start:
                        if isinstance(module[1], nn.BatchNorm2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
                            module[1].running_mean.copy_(state_dict[(module[0] + ".running_mean")])
                            module[1].running_var.copy_(state_dict[(module[0] + ".running_var")])
                            module[1].num_batches_tracked.copy_(state_dict[(module[0] + ".num_batches_tracked")])
                        elif isinstance(module[1], nn.Conv2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                        elif isinstance(module[1], nn.Linear):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
    
    ### A bit cluttered, but this is used to ensure that frozen weights don't change due to decay
    def load_frozen(self, state_dict):
        with torch.no_grad():
            if self.args.frozen_subnet == "fa":
                for name, module in enumerate(self.model.named_modules()):
                    if name < self.f_b_start:
                        if isinstance(module[1], nn.BatchNorm2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
                            module[1].running_mean.copy_(state_dict[(module[0] + ".running_mean")])
                            module[1].running_var.copy_(state_dict[(module[0] + ".running_var")])
                            module[1].num_batches_tracked.copy_(state_dict[(module[0] + ".num_batches_tracked")])
                        elif isinstance(module[1], nn.Conv2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                        elif isinstance(module[1], nn.Linear):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
            elif self.args.frozen_subnet == "fb":
                for name, module in enumerate(self.model.named_modules()):
                    if name >= self.f_b_start and self.f_b_start != 0:
                        if isinstance(module[1], nn.BatchNorm2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])
                            module[1].running_mean.copy_(state_dict[(module[0] + ".running_mean")])
                            module[1].running_var.copy_(state_dict[(module[0] + ".running_var")])
                            module[1].num_batches_tracked.copy_(state_dict[(module[0] + ".num_batches_tracked")])
                        elif isinstance(module[1], nn.Conv2d):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                        elif isinstance(module[1], nn.Linear):
                            module[1].weight.copy_(state_dict[(module[0] + ".weight")])
                            module[1].bias.copy_(state_dict[(module[0] + ".bias")])


    ### Freeze the appropriate subnetwork by setting requires_grad=False. Used along with reloading of weights to ensure fixed states
    def set_no_grad(self):
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
                