# SemirobustSubnetworks
Code for transfer of robustness from semirobust subnetwork
This repository provides the code for evaluating the transfer of robustness from semirobust subnetwork. This includes the code for evaluating the linear dependencies as well as the non-linear dependencies calculated by Mutual Information (MI). The MI code includes both the standard adversarial training setup used for the primary experiments, as well as the implementation of benchmark methods ATTA (https://github.com/haizhongzheng/ATTA?tab=readme-ov-file) and TRADES (https://github.com/yaodongyu/TRADES). For TRADES we use the implementation provided by ATTA.  

The MI code can be run with VGG16, Resnet-18, and WideResnet-34-10 on CIFAR-10, CIFAR-100, and Tiny Imagenet. The CIFAR datasets will be downloaded while running the code, but for the more complicated Tiny Imagenet setup we provide an ipynb file for doing so manually which was modified from: https://github.com/rcamino/pytorch-notebooks/blob/master/Train%20Torchvision%20Models%20with%20Tiny%20ImageNet-200.ipynb

The code can be run using the main scripts as follows:

Algorithm1.py: Runs Algorithm 1 for a given choice of attack, network, dataset, and number of layers in subnetwork fb

Trial Rhos.py: Records rho values and layer sensitivities (defined in https://arxiv.org/pdf/1909.06978.pdf) for each epoch during one trial of Algorithm 1

Subnetwork Training.py: Used to pretrain the normal and adversarially robust models (fa,fb) and(fa*,fb*) for Algorithm 1

Note: The benchmarks use their own modified versions of these files. 

AlgorithmS1.py: Runs the linear dependency Algorithm S1 from the Appendix.

We provide ipynb files for running sample experiments for each script
