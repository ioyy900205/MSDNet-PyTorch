'''
Date: 2021-07-08 11:57:02
LastEditors: Liuliang
LastEditTime: 2021-07-14 09:44:15
Description: liangnet
'''

import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F
import torch
import math
import pdb
from torch.nn.modules.module import Module


# ================================================================== #
#                说明：sample进来之后的底层处理层                                             
# ================================================================== #	
class BottomLayer(nn.Module):
    def __init__(self):
        super(BottomLayer,self).__init__()
        self.out_channel = 64
        
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
        
# ================================================================== #
#                说明：Resnet基础块                                             
# ================================================================== #
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)        
        return out

class LiangNet(nn.Module):
    def __init__(self,
    block,
    block_nums,
    num_classes=1000):
        super(LiangNet,self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = 5        
        self.in_channel = 64
        self.out_channel = self.in_channel
        self.stide = 1
       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # ================================================================== #
        #                说明：blocks                                             
        # ================================================================== #	
        for i in range(self.nBlocks):
            print(f'***********block{i}***********')
            m = self._build_block(block,block_nums,i)
            self.blocks.append(m)
        
        # ================================================================== #
        #                说明：classifiers                                             
        # ================================================================== #	
        for i in range(self.nBlocks):
            layers = []
            if i == self.nBlocks-1:
                print(f'***********classifier{i}***********')
                self.classifier.append(nn.AdaptiveAvgPool2d((1, 1)))
                # self.classifier.append(torch.flatten(torch.tensor(2048),1))
                self.classifier.append(nn.Linear(2048, num_classes))
                
                print(f'the classifier is competed')       
        

    def _build_block(self,block,block_nums,i):
        print(f'the {i}th block is building')        
        layers = []

        if i == 0 :
            layers = [BottomLayer()] 
            print(f'the {i}th block is complished')
            
        else:            
            for element in range(block_nums[i-1]): #每层block有重复的block_nums[i-1]个模块
                
                
                print(f'the {element}th element in block{i}')
                 
                stride = 2 if element == 0 and i >= 2 else 1
                


                downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channel * block.expansion)) if element == 0 else None
               
                layers.append(
                    block(self.in_channel,
                    self.out_channel,
                    stride = stride,
                    downsample = downsample)
                    )
                    
                if element == 0: 
                    self.in_channel = self.out_channel*block.expansion
                    

            if element == (block_nums[i-1] -1): self.out_channel = self.out_channel * 2




            
        return nn.Sequential(*layers)

    def forward(self,x):
        result = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            # x = self.classifier[i](x)
            
            if i == (self.nBlocks-1):
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)                
                result.append(x)       
        return result


def liangnet(num_classes=1000):
    
    return LiangNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

if __name__=='__main__':


    model = liangnet()
    input = torch.randn(1,3,224,224)
    out = model(input)
    print(model)
    # print(out[0].shape)
    # c = out[0].view(-1)
    # print(c)
    # print(out[1].shape)
