"""   
    Version of Resnet which incorportates a unique linear combination layer 
    that stores lambda values to make predicted outputs as a linear combination of activations in f_a^*  
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import tensorflow as tf
import numpy as np
import time


class LinCom(nn.Module):
    """ Custom Linear layer which computes a linear combination of all convolution layer activations in f_a^* """
    def __init__(self, model, fb_len, batchsize):
        super().__init__()
        self.batchsize=batchsize
        self.lambdas = []
        self.xs = []
        ### These are the dimensions of flattening the feature map over N channels at each conv layer
        if model == "ResNet18":
            dims = [65536, 16384, 16384, 16384, 16384, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 4096, 2048, 2048, 2048, 2048, 512]
        self.dim_out = 10
        ### Calculate how many layers are included in f_a^* based on the size of f_b^*
        self.num_inputs = len(dims)-fb_len
        
        ### lambdas form a matrix which maps from the activation matrix of f_a^* to the output classes (dim_out=10)
        self.lambdas = np.random.normal(loc=0.0, scale=1.0, size=(np.sum(dims[:self.num_inputs]),self.dim_out))
        ### store initialize lambdas for future reseting, acts as a fixed seed  
        self.random_lambdas = self.lambdas
        print("lambdas shape: ", self.lambdas.shape)

    ### Reset any calculated and updated lambdas
    def reset_lambdas(self):
      self.lambdas = self.random_lambdas

    ### Reset any stored f_a^* activations
    def reset_xs(self):
      self.xs = []

    def forward(self, x_list):
        ### Get the activations of all layers included in f_a^*
        templist = x_list[:self.num_inputs]
        x = torch.cat(templist[:],dim=1)
        ### Append activations to the stored set for later lambda updates
        self.xs.append(x)

        ### Get the output values by multiplying the activations in f_a^* by their corresponding lambdas
        acts = torch.mm(x.double(), torch.from_numpy(self.lambdas).cuda().double())

        return acts



    


#########################################################################################################################################################
#Resnet
#########################################################################################################################################################


# Source: https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        args,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        ### Lincom parameters
        self.lincom = False
        self.batchsize = args.batchsize
        self.fb_len = args.num_fb_layers


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lincom1 = LinCom(model="ResNet18", fb_len=self.fb_len, batchsize=self.batchsize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.lincom == False:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            return x
        ### If set to use linear combinations, we expand out the forward call across each individual layer in order to retain intermediate layer outputs
        ###     which are then used in calculating the network outputs through the lincom layer
        else:
            x0 = self.conv1(x)
            x1 = self.bn1(x0)
            x1 = self.relu(x1)
            x1 = self.maxpool(x1)

            ### Layer 1
            identity = x1
            x1 = self.layer1[0].conv1(x1)
            x2 = self.layer1[0].bn1(x1)
            x2 = self.layer1[0].relu(x2)
            x2 = self.layer1[0].conv2(x2)
            x3 = self.layer1[0].bn2(x2)
            x3 += identity
            x3 = self.layer1[0].relu(x3)

            identity = x3
            x3 = self.layer1[1].conv1(x3)
            x4 = self.layer1[1].bn1(x3)
            x4 = self.layer1[1].relu(x4)
            x4 = self.layer1[1].conv2(x4)
            x5 = self.layer1[1].bn2(x4)
            x5 += identity
            x5 = self.layer1[1].relu(x5)


            ### Layer 2
            identity = x5
            x5 = self.layer2[0].conv1(x5)
            x6 = self.layer2[0].bn1(x5)
            x6 = self.layer2[0].relu(x6)
            x6 = self.layer2[0].conv2(x6)
            x7 = self.layer2[0].bn2(x6)
            temp = self.layer2[0].downsample[0](identity)
            identity = self.layer2[0].downsample[1](temp)
            x7 += identity
            x7 = self.layer2[0].relu(x7)

            identity = x7
            x7 = self.layer2[1].conv1(x7)
            x8 = self.layer2[1].bn1(x7)
            x8 = self.layer2[1].relu(x8)
            x8 = self.layer2[1].conv2(x8)
            x9 = self.layer2[1].bn2(x8)
            x9 += identity
            x9 = self.layer2[1].relu(x9)


            ### Layer 3
            identity = x9
            x9 = self.layer3[0].conv1(x9)
            x10 = self.layer3[0].bn1(x9)
            x10 = self.layer3[0].relu(x10)
            x10 = self.layer3[0].conv2(x10)
            x11 = self.layer3[0].bn2(x10)
            temp = self.layer3[0].downsample[0](identity)
            identity = self.layer3[0].downsample[1](temp)
            x11 += identity
            x11 = self.layer3[0].relu(x11)

            identity = x11
            x11 = self.layer3[1].conv1(x11)
            x12 = self.layer3[1].bn1(x11)
            x12 = self.layer3[1].relu(x12)
            x12 = self.layer3[1].conv2(x12)
            x13 = self.layer3[1].bn2(x12)
            x13 += identity
            x13 = self.layer3[1].relu(x13)


            ### Layer 4
            identity = x13
            x13 = self.layer4[0].conv1(x13)
            x14 = self.layer4[0].bn1(x13)
            x14 = self.layer4[0].relu(x14)
            x14 = self.layer4[0].conv2(x14)
            x15 = self.layer4[0].bn2(x14)
            temp = self.layer4[0].downsample[0](identity)
            identity = self.layer4[0].downsample[1](temp)
            x15 += identity
            x15 = self.layer4[0].relu(x15)

            identity = x15
            x15 = self.layer4[1].conv1(x15)
            x16 = self.layer4[1].bn1(x15)
            x16 = self.layer4[1].relu(x16)
            x16 = self.layer4[1].conv2(x16)
            x17 = self.layer4[1].bn2(x16)
            x17 += identity
            x17 = self.layer4[1].relu(x17)
            
            
            x17 = self.avgpool(x17)
            x17 = x17.reshape(x17.size(0), -1)
            
            ### Pass all intermediate convolutional layer outputs to lincom layer, which will apply lambda to those in f_a^*
            fb = self.lincom1(x_list=[x0.reshape(x0.size()[0],-1),
                                    x1.reshape(x1.size()[0],-1),
                                    x2.reshape(x2.size()[0],-1),
                                    x3.reshape(x3.size()[0],-1),
                                    x4.reshape(x4.size()[0],-1),
                                    x5.reshape(x5.size()[0],-1),
                                    x6.reshape(x6.size()[0],-1),
                                    x7.reshape(x7.size()[0],-1),
                                    x8.reshape(x8.size()[0],-1),
                                    x9.reshape(x9.size()[0],-1),
                                    x10.reshape(x10.size()[0],-1),
                                    x11.reshape(x11.size()[0],-1),
                                    x12.reshape(x12.size()[0],-1),
                                    x13.reshape(x13.size()[0],-1),
                                    x14.reshape(x14.size()[0],-1),
                                    x15.reshape(x15.size()[0],-1),
                                    x16.reshape(x16.size()[0],-1),
                                    x17])
                                    

            return fb



def _resnet(args, arch, block, layers, progress, device, **kwargs):
    print("Start")
    model = ResNet(args, block, layers, num_classes=10, **kwargs)
    return model


def resnet18(args, progress=True, device="cpu", **kwargs):
    return _resnet(
        args, "resnet18", BasicBlock, [2, 2, 2, 2], progress, device, **kwargs
    )