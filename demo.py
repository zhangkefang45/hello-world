#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import random

# 超参数
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 100     # 记忆库大小
N_ACTIONS = 7               # 机械臂能做的动作
N_STATES = 224*224*4


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.dense121 = models.densenet121(True)
        self.fc1 = nn.Linear(2000 + 3, 2048)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 3)
        # self.fc3.weight.data *= 10

    def forward(self, rgb, deep, joint):
        # rgb = self.dense121(rgb)
        # deep = self.dense121(deep)
        # # x = x.view(-1, 48*4*4)
        # x = torch.cat([rgb.float(), deep.float(), joint.float()], dim=1)
        # a = self.fc1(x)
        # x = F.relu(self.fc2(a))
        # x = self.fc3(x)
        x = self.dense121(rgb)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    a = Net().cuda()
    joint, rgb, deep = torch.randn(1, 3).cuda(), torch.randn(1, 3, 224, 224).cuda(), torch.randn(1, 3, 224, 224).cuda()
    out = a.forward(rgb.float(), deep.float(), joint.float())
    print(out.shape)




