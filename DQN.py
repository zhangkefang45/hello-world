#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import os
import random

# 超参数
BATCH_SIZE = 60
LR = 0.001                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000     # 记忆库大小
N_ACTIONS = 7               # 机械臂能做的动作
N_STATES = 224*224*4


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.dense121 = models.resnet50(True)  #(1, 1000)
        self.fc1 = nn.Linear(2000 + 3, 2048)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 3)
        # self.fc3.weight.data *= 10

    def forward(self, rgb, deep, joint):
        rgb = self.dense121(rgb)
        deep = self.dense121(deep)
        # x = x.view(-1, 48*4*4)
        x = torch.cat([rgb.float(), deep.float(), joint.float()], dim=1)
        a = self.fc1(x)
        x = F.relu(self.fc2(a))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DQN(object):
    def __init__(self):
        self.device_ids = [0, 1]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_net = Net().cuda(self.device_ids[0])  # .to(self.device)
        self.target_net = Net().cuda(self.device_ids[0])
        self.eval_net = nn.DataParallel(self.eval_net, device_ids=self.device_ids)
        self.target_net = nn.DataParallel(self.target_net, device_ids=self.device_ids)
        self.learn_step_counter = 0     # 用于target更新计时
        self.memory_counter = 0         # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, (224*224*4+3)*2+4))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch1 的优化器
        self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.device_ids)
        self.loss_func = nn.MSELoss()  # 误差公式

    # 根据神经网络选取一个值


    def choose_action(self, x):
        a = np.random.randint(1, 1000)
        # if a > 10:
        if np.random.uniform() < EPSILON*(math.exp(self.memory_counter-5000) if self.memory_counter <= 5000 else 1):
            # print "****action from net****"
            joint_view, image_view = x
            image_view = image_view / (256 * 256)
            image_view = image_view.astype(np.float32)
            rgb_np = np.array(image_view).reshape(-1, 224, 224, 4)[:, :, :, :3]
            dep_np = np.array(image_view).reshape(-1, 224, 224, 4)[:, :, :, 3].reshape(-1, 224, 224, 1)
            dep_np = np.concatenate((dep_np, dep_np, dep_np), axis=3)
            image_view_rgb = torch.from_numpy(rgb_np)
            image_view_rgb = image_view_rgb.permute(0, 3, 1, 2).cuda()
            image_view_dep = torch.from_numpy(dep_np)
            image_view_dep = image_view_dep.permute(0, 3, 1, 2).cuda()
            joint_view = torch.from_numpy(np.array(joint_view).reshape(-1, 3)).cuda()
            action = self.eval_net.forward(image_view_rgb, image_view_dep, joint_view).detach()
            action = action.cpu().numpy()
        else:
            # print "****action for rand****"
            action = np.random.uniform(low=-0.2, high=0.2, size=3)
            action = action[np.newaxis, :]
        return action

    def store_transition(self, s, a, r, s_):
        a = np.array(a).reshape(-1, 3)
        if a[0][0] is np.nan:
            return
        s1, s2 = s
        s3, s4 = s_

        if str(type(s3)) == '<type \'numpy.float64\'>':
            s_ = s
        #  s3 == list == numpy.float todo
        s3, s4 = s_
        s1 = np.array(s1).reshape(-1, 3)
        s2 = np.array(s2).reshape(-1, 224*224*4)
        r = np.array(r).reshape(-1, 1)
        s3 = np.array(s3).reshape(-1, 3)
        s4 = np.array(s4).reshape(-1, 224*224*4)

        transition = np.hstack((s1, s2, a, r, s3, s4))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # type: () -> object
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_joint1 = torch.FloatTensor((b_memory[:, :3]).reshape(-1, 3)).cuda()
        rgb_np = (b_memory[:, 3:N_STATES + 3]).reshape(-1, 224, 224, 4)[:, :, :, :3]
        dep_np = (b_memory[:, 3:N_STATES + 3]).reshape(-1, 224, 224, 4)[:, :, :, 3].reshape(-1, 224, 224, 1)
        dep_np = np.concatenate((dep_np, dep_np, dep_np), axis=3)
        b_rgb2 = torch.FloatTensor(rgb_np).permute(0, 3, 1, 2).cuda()
        b_dep2 = torch.FloatTensor(dep_np).permute(0, 3, 1, 2).cuda()
        # b_s = b_s1, b_s2
        b_a = torch.LongTensor((b_memory[:, N_STATES+3:N_STATES + 6]).reshape(-1, 3).astype(float)).cuda()
        b_r = torch.FloatTensor((b_memory[:, N_STATES + 6:N_STATES + 7]).reshape(-1, 1)).cuda()
        b_joint_1 = torch.FloatTensor((b_memory[:, N_STATES + 7:N_STATES + 10]).reshape(-1, 3)).cuda()
        rgb_np_ = (b_memory[:, -N_STATES:]).reshape(-1, 224, 224, 4)[:, :, :, :3]
        dep_np_ = (b_memory[:, -N_STATES:]).reshape(-1, 224, 224, 4)[:, :, :, 3].reshape(-1, 224, 224, 1)
        dep_np_ = np.concatenate((dep_np_, dep_np_, dep_np_), axis=3)
        b_rgb_2 = torch.FloatTensor(rgb_np_).permute(0, 3, 1, 2).cuda()
        b_dep_2 = torch.FloatTensor(dep_np_).permute(0, 3, 1, 2).cuda()

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_rgb2, b_dep2, b_joint1)              # shape (batch, 1) picture and joint
        q_next = self.target_net(b_rgb_2, b_dep_2, b_joint_1).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()
        self.optimizer.module.step()

    def save_model(self):
        model_number = self.get_file_number("eval_dqn")
        print("model_number:",model_number)
        torch.save(self.eval_net, "model/eval_dqn/" + str(model_number) + ".pkl")
        torch.save(self.target_net, "model/target_dqn/" + str(model_number) + ".pkl")

    def get_file_number(self, dir_name):
        a = 0
        file_dir = "/home/ljt/ws/src/fetch_moveit_config/model/"
        for root, dirs, files in os.walk(file_dir+dir_name):
            a = len(files)
        return a

    def get_last_model(self, dir_name):
        a = 0
        file_dir = "/home/ljt/ws/src/fetch_moveit_config/model/"
        lists = os.listdir(file_dir+dir_name)  # 列出目录的下所有文件和文件夹保存到lists
        lists.sort(key=lambda fn: os.path.getmtime(file_dir+dir_name + "/" + fn))  # 按时间排序
        return int(lists[-1][:-4])


if __name__ == "__main__":
    net = DQN()
    # print net.get_file_number("target_dqn")
    # print net.get_last_model("target_dqn")

    # input = torch.randn(3), torch.randn(4, 224, 224)
    # input = np.random.randn(3), np.random.randn(4, 224, 224)
    # out = net.choose_action(input)
    # # net.learn()
    # print(type(out))



