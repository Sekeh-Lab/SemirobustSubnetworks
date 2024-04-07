"""Main entry point for doing all pruning-related stuff. Adapted from https://github.com/arunmallya/packnet/blob/master/src/main.py"""
from __future__ import division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import argparse
import pickle
import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchnet as tnt

import torchattacks
# from autoattack.autoattack import AutoAttack

from AuxiliaryScripts import mi_estimator, utils, adaptive_data_aug
from AuxiliaryScripts.utils import *
from AuxiliaryScripts.adaptive_data_aug import atta_aug, atta_aug_trans, inverse_atta_aug

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

    def __init__(self, args, model, x_nat, x_adv, y_train, testloader, f_b_start, root_save_path, starting_dict):
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.f_b_start = f_b_start
        self.root_save_path = root_save_path
        self.starting_dict = starting_dict
        
        self.x_nat = x_nat
        self.x_adv = x_adv
        self.y_train = y_train

        print("Checking sum of X_nat: ", torch.sum(self.x_nat), flush=True)

        ### A backup of the inputs to reset perturbations with for ATTA
        self.x_backup = copy.deepcopy(x_nat)
        self.batch_count = math.ceil(len(x_nat) / args.batchsize)


        self.test_data_loader = testloader
        self.x_test_adv = None
        self.y_test_adv = None
        # self.preds_test_adv = None


        self.criterion = nn.CrossEntropyLoss()

        ### Explicitly set frozen layers requires_grad=False. 
        if self.args.frozen_subnet != "none":
            self.set_no_grad()
        # self.set_train_sub()
        self.model.train()
        
    
    #!# We don't use the ATTA method for evaluation as the test data is "unseen" by the network so can't accumulate perturbations over epochs
    ### For this reason we also hard-code the evaluation to use 10 attack iterations so that it's not influenced by the number we use for defense
    def eval(self, biases=None):
        """Performs evaluation."""
        self.model.eval()
        error_meter_norm = None
        error_meter = None
        ### Ensure you only ever keep the most recent adv test data for use in calculating MI
        self.x_test_norm = None
        self.y_test_norm = None
        self.x_test_adv = None
        self.y_test_adv = None
        # self.preds_test_adv = None

        print('Performing eval...', flush=True)
        
        ### For each batch we get the normal accuracy and the adversarial accuracy if applying an attack
        for batch, label in self.test_data_loader:
            if self.cuda:
                batch = batch.cuda()
                label = label.cuda()
        
            # print("Clean eval")
            with torch.no_grad():
                output = self.model(batch)


            # Init error meter.
            if error_meter_norm is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter_norm = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter_norm.add(output.data, label)

                
            ### Setting current "attacked" data for consistency with attack code 
            if self.args.attacktype == "none":
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


            elif self.args.attacktype != "none":
                if self.args.attacktype == "fgsm":
                    attack = torchattacks.FGSM(self.model, eps=self.args.eps)
                elif self.args.attacktype == "pgd":
                    attack = torchattacks.PGD(self.model, eps=self.args.eps, alpha=self.args.eps_step, steps=10, random_start=True)
                elif self.args.attacktype == "autoattack":
                    if self.args.dataset == "CIFAR100":
                        attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=100, seed=None, verbose=False)
                    elif self.args.dataset == "TinyImagenet":
                        attack = torchattacks.AutoAttack(self.model, norm='Linf', eps=self.args.eps, version='standard', n_classes=200, seed=None, verbose=False)
                    else:
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


        errors_norm = error_meter_norm.value()
        print('Error Norm: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors_norm)))
        
        self.model.train()

        if self.args.attacktype != "none":
            errors = error_meter.value()
            print('Error: ' + ', '.join('@%s=%.2f' %
                                        t for t in zip(topk, errors)))                                    
            return errors_norm,errors
        else:
            return errors_norm,errors_norm





    #!# Modified train method leveraging ATTA and/or TRADES
    def train_epoch_nonadv(self, optimizer):
        print("! Doing non-adversarial Training batch", flush=True)
        """Runs model for each batch."""
        for idx in range(self.batch_count):
            if idx+1 != self.batch_count:
                x_batch_nat = self.x_nat[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
                x_batch_adv = self.x_adv[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
                y_batch =   self.y_train[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
            else:
                x_batch_nat = self.x_nat[idx*self.args.batchsize:]
                x_batch_adv = self.x_adv[idx*self.args.batchsize:]
                y_batch =   self.y_train[idx*self.args.batchsize:]

            if self.cuda:
                x_batch_nat = x_batch_nat.cuda()
                y_batch = y_batch.cuda()
            x_batch_nat = Variable(x_batch_nat, requires_grad=True)
            y_batch = Variable(y_batch)

            # Set grads to 0.
            optimizer.zero_grad()
            # Do forward-backward.
            output = self.model(x_batch_nat)

            self.criterion(output, y_batch).backward()
            # Update params.
            optimizer.step()
            self.load_frozen(self.starting_dict)




    #!# Modified train method leveraging ATTA and/or TRADES
    def train_epoch_adv(self, optimizer):
        """Runs model for each batch."""

        for idx in range(self.batch_count):
            if idx+1 != self.batch_count:
                x_batch_nat = self.x_nat[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
                x_batch_adv = self.x_adv[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
                y_batch =   self.y_train[idx*self.args.batchsize:(idx+1)*self.args.batchsize]
            else:
                x_batch_nat = self.x_nat[idx*self.args.batchsize:]
                x_batch_adv = self.x_adv[idx*self.args.batchsize:]
                y_batch =   self.y_train[idx*self.args.batchsize:]

            if self.cuda:
                x_batch_nat = x_batch_nat.cuda()
                x_batch_adv = x_batch_adv.cuda()
                y_batch = y_batch.cuda()

            x_batch_nat = Variable(x_batch_nat, requires_grad=True)
            y_batch = Variable(y_batch)


            #atta-aug
            rst = torch.zeros(x_batch_adv.size(0),3,32,32).cuda()
            x_batch_adv, transform_info = atta_aug(x_batch_adv, rst)
            rst = torch.zeros(x_batch_nat.size(0),3,32,32).cuda()
            x_batch_nat = atta_aug_trans(x_batch_nat, transform_info, rst)

            x_adv_next = utils.get_adv_atta(
                               model=self.model,
                               x_natural=x_batch_nat,
                               x_adv=x_batch_adv,
                               y=y_batch,
                               step_size=self.args.eps_step,
                               epsilon=self.args.eps,
                               num_steps=self.args.attack_iterations,
                               loss_type=self.args.loss_type
            )

            x_adv_next = Variable(x_adv_next, requires_grad=False)
            optimizer.zero_grad()  


            if self.args.loss_type == "mat":
                criterion_ce = nn.CrossEntropyLoss()
                loss = (1.0 / self.args.batchsize) * criterion_ce(F.log_softmax(self.model(x_adv_next), dim=1), y_batch)
            elif self.args.loss_type == "trades":
                criterion_kl = nn.KLDivLoss(size_average=False)
                nat_logits = self.model(x_batch_nat)
                loss_natural = F.cross_entropy(nat_logits, y_batch)
                loss_robust = (1.0 / self.args.batchsize) * criterion_kl(F.log_softmax(self.model(x_adv_next), dim=1),F.softmax(nat_logits, dim=1))
                loss = loss_natural + self.args.beta * loss_robust
            else:
                print("Unknown loss method.")
                raise


            # Update params.
            loss.backward()
            optimizer.step()
            self.load_frozen(self.starting_dict)

            #!# Confirm that the perturbation is applying to self.x_adv
            if idx+1 != self.batch_count:
                # self.x_adv[idx*self.args.batchsize:(idx+1)*self.args.batchsize] = x_adv_next
                self.x_adv[idx*self.args.batchsize:(idx+1)*self.args.batchsize] =  inverse_atta_aug(self.x_adv[idx*self.args.batchsize:(idx+1)*self.args.batchsize], x_adv_next, transform_info)
            else:
                # self.x_adv[idx*self.args.batchsize:] = x_adv_next
                self.x_adv[idx*self.args.batchsize:] =  inverse_atta_aug(self.x_adv[idx*self.args.batchsize:], x_adv_next, transform_info)


    









    
    def train(self, epochs, optimizer, scheduler=None, save=True, target_accuracy=0, delta=0, eval_all=False):
        """Performs training."""

        best_test_accuracy = 0
        os.makedirs(self.root_save_path, exist_ok=True)    
        checkpoint_path = (self.root_save_path + "checkpoint_" + str(self.args.num_fb_layers))

        if self.args.cuda:
            self.model = self.model.cuda()

        ### Explicitly set frozen layers requires_grad=False. 
        if self.args.frozen_subnet != "none":
            self.set_no_grad()
        self.model.train()

        for idx in range(epochs):
            print('Epoch: %d' % (idx), "Learning rate:", optimizer.param_groups[0]['lr'], flush=True)

            ### shuffle the training data in tandem prior to the epoch training
            self.x_nat, self.x_adv, self.y_train = utils.shuffle_data(self.x_nat, self.x_adv, self.y_train)

            #!# Following ATTA we reset perturbations every 10 epochs for ATTA and ATTA-Trades and every 1 epochs for TRADES and regular PGD-10 training this comparison experiment
            if idx % self.args.epochs_reset == 0:
                print("Reset perturbations", flush=True)
                print("Pre-reset sum of X_nat: ", torch.sum(self.x_nat), flush=True)
                print("Pre-reset sum of X_adv: ", torch.sum(self.x_adv), flush=True)
                self.x_adv = copy.deepcopy(self.x_nat)
                self.x_adv = self.x_adv.detach() + 0.001 * torch.randn(self.x_adv.shape).detach()
                print("Checking sum of X_nat: ", torch.sum(self.x_nat), flush=True)
                print("Checking sum of X_adv: ", torch.sum(self.x_adv), flush=True)
                print("Checking order of labels y_train: ", self.y_train[:10], flush=True)

            ### Train over all batches for the current epoch
            if self.args.defensetype != "none":
                self.train_epoch_adv(optimizer)
            else:
                self.train_epoch_nonadv(optimizer)


            if scheduler is not None:
                scheduler.step()








            ### Saves some training time during full-network training by not generating attacks on test data each epoch.
            if idx % 5 == 0 or eval_all == True or (idx+1 == epochs): 
                test_errors_norm, test_errors = self.eval()
                test_accuracy_norm = 100 - test_errors_norm[0]  # Top-1 accuracy.
                test_accuracy = 100 - test_errors[0]  # Top-1 accuracy.

                print('Test Normal Accuracy: %0.2f%%' % (test_accuracy_norm))
                print('Test Adversarial Accuracy: %0.2f%%' % (test_accuracy))
                      
                ### Track best accuracy based on adversarial accuracy
                if test_accuracy >= best_test_accuracy:
                    best_test_accuracy=test_accuracy
                
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
        # self.save_model(checkpoint_path)
        return best_test_accuracy, 0
    
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def save_model(self, path):
        # print("Saving model to path: ", path)
        torch.save(self.model.state_dict(), path)
       
    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
            
        
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
            # else:
            #     print("No frozen layers to load")
    


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
                


    def get_counts(self):
        print(self.y_test_adv.size())
        y_test = self.y_test_adv.numpy()
        print(self.y_test_adv.size())
        if self.args.dataset=="CIFAR100":
            self.x_split = [self.x_test_adv[np.where(y_test[:] == j)] for j in range(100)]                                   
            self.y_split = [self.y_test_adv[np.where(y_test[:] == j)] for j in range(100)]                                   
            self.label_counts = np.bincount(y_test[:], minlength=100) / len(y_test)                                     
        elif self.args.dataset == "CIFAR10":
            self.x_split = [self.x_test_adv[np.where(y_test[:] == j)] for j in range(10)]
            self.y_split = [self.y_test_adv[np.where(y_test[:] == j)] for j in range(10)]                                   
            self.label_counts = np.bincount(y_test[:], minlength=10) / len(y_test)
        elif self.args.dataset == "TinyImagenet":
            self.x_split = [self.x_test_adv[np.where(y_test[:] == j)] for j in range(200)]
            self.y_split = [self.y_test_adv[np.where(y_test[:] == j)] for j in range(200)]                                   
            self.label_counts = np.bincount(y_test[:], minlength=200) / len(y_test)
        

    
    def get_mi_estimate(self, parent_index, child_index):
        # Use outputs of f^(j-3) to f^(j)
        mi_est = 0
        if self.args.dataset == "CIFAR10":
            n = 10
        elif self.args.dataset == "CIFAR100":
            n = 100
        elif self.args.dataset == "TinyImagenet":
            n = 200
      

        self.model.eval()
        for i in range(n):
            # acts, _ = utils.activations(self.x_split[i], self.y_split[i], self.model, self.args.cuda)
            # act_parent = acts[parent_index].cpu().numpy()
            # act_child = acts[child_index].cpu().numpy() 

            act_parent = activations_single_layer(self.x_split[i], self.model, self.args.cuda, parent_index)
            act_child = activations_single_layer(self.x_split[i], self.model, self.args.cuda, child_index)
            
            if len(act_parent.shape) > 2:                                                                     
                act_parent = np.reshape(act_parent, (np.shape(act_parent)[0], -1))
            if len(act_child.shape) > 2:                                                                     
                act_child = np.reshape(act_child, (np.shape(act_child)[0], -1))
              
            mi_est += self.label_counts[i] * mi_estimator.EDGE(act_parent, act_child,                        
              normalize_epsilon=False, L_ensemble=1, stochastic=True)

        self.model.train()
        return mi_est



    def get_activations(self, saveID):
        sensitivity = {}
        self.model.eval()
        norm_acts, adv_acts = {}, {}
        norm_acts["x"], norm_acts["y"] = utils.activations(self.x_test_norm, self.y_test_norm, self.model, self.args.cuda, mean=False)
        adv_acts["x"], adv_acts["y"] = utils.activations(self.x_test_adv, self.y_test_adv, self.model, self.args.cuda, mean=False)
        print("\nDone collecting activations")


        for key in norm_acts['x'].keys():
            if len(norm_acts["x"][key].shape) > 3:
                actsdif = torch.linalg.matrix_norm((norm_acts["x"][key] - adv_acts["x"][key]), ord=1)
                filternorms = torch.linalg.matrix_norm(norm_acts["x"][key], ord=1)
                normdifs = actsdif/filternorms
                print("Normalized differences shape for filter activations: ", normdifs.size())
                ### Get the mean for all filters, then for all images
                normdifs = torch.mean(normdifs,dim=1)
                sensitivity[key] = torch.mean(normdifs)

            else:
                actsdif = torch.abs((norm_acts["x"][key] - adv_acts["x"][key]))
                neuronnorms = torch.abs(norm_acts["x"][key])
                normdifs = actsdif/neuronnorms
                print("Normalized differences shape for neuron activations: ", normdifs.size())
                ### Get the mean for all neurons, then for all images
                normdifs = torch.mean(normdifs,dim=1)
                sensitivity[key] = torch.mean(normdifs)

        print("sensitivities: ", sensitivity)

        self.model.train()
        return sensitivity
