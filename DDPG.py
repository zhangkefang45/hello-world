#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
import math
import random
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 10000
MAX_EP_STEPS = 20
LR_A = 0.0005    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 18
N_STATES = 224*224*4
RENDER = False
EPSILON = 0.9
###############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet,self).__init__()
        self.dense121 = models.resnet50(True)  # (1, 1000)
        self.fc1 = nn.Linear(2000 + 3, 2048)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, rgb, deep, joint):
        rgb = self.dense121(rgb)
        deep = self.dense121(deep)
        # x = x.view(-1, 48*4*4)
        x = torch.cat([rgb.float(), deep.float(), joint.float()], dim=1)
        a = self.fc1(x)
        x = F.relu(self.fc2(a))
        x = self.fc3(x)
        return x

class CNet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(CNet,self).__init__()
        self.dense121 = models.resnet50(True)  # (1, 1000)
        self.fc1 = nn.Linear(2000 + 6, 2048)
        self.fc2 = nn.Linear(2048, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, rgb, deep, joint, ac):
        rgb = self.dense121(rgb)
        deep = self.dense121(deep)
        # x = x.view(-1, 48*4*4)
        x = torch.cat([rgb.float(), deep.float(), joint.float(), ac.float()], dim=1)
        a = self.fc1(x)
        x = F.relu(self.fc2(a))
        x = self.fc3(x)
        return x


class DDPG(object):
    def __init__(self):
        self.device_ids = [0, 1]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = np.zeros((MEMORY_CAPACITY, (224 * 224 * 4 + 3) * 2 + 4))
        self.memory_counter = 0  # 记忆库计数
        #self.sess = tf.Session()
        self.Actor_eval = ANet().cuda(self.device_ids[0])
        self.Actor_target = ANet().cuda(self.device_ids[0])
        self.Critic_eval = CNet().cuda(self.device_ids[0])
        self.Critic_target = CNet().cuda(self.device_ids[0])
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        # print "****action from net****"
        # if np.random.uniform() < EPSILON * (math.exp(self.memory_counter - 5000) if self.memory_counter <= 5000 else 1):
        joint_view, image_view = s
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
        action = self.Actor_eval.forward(image_view_rgb, image_view_dep, joint_view).detach()
        action = action.cpu().numpy()
        # else:
        #     # print "****action for rand****"
        #     action = np.random.uniform(low=-0.1, high=0.1, size=3)
        #     action = action[np.newaxis, :]
        return action

    def learn(self):

        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_joint1 = torch.FloatTensor((b_memory[:, :3]).reshape(-1, 3)).cuda()
        rgb_np = (b_memory[:, 3:N_STATES + 3]).reshape(-1, 224, 224, 4)[:, :, :, :3]
        dep_np = (b_memory[:, 3:N_STATES + 3]).reshape(-1, 224, 224, 4)[:, :, :, 3].reshape(-1, 224, 224, 1)
        dep_np = np.concatenate((dep_np, dep_np, dep_np), axis=3)
        b_rgb2 = torch.FloatTensor(rgb_np).permute(0, 3, 1, 2).cuda()
        b_dep2 = torch.FloatTensor(dep_np).permute(0, 3, 1, 2).cuda()
        # b_s = b_s1, b_s2
        b_a = torch.LongTensor((b_memory[:, N_STATES + 3:N_STATES + 6]).reshape(-1, 3).astype(float)).cuda()
        b_r = torch.FloatTensor((b_memory[:, N_STATES + 6:N_STATES + 7]).reshape(-1, 1)).cuda()
        b_joint_1 = torch.FloatTensor((b_memory[:, N_STATES + 7:N_STATES + 10]).reshape(-1, 3)).cuda()
        rgb_np_ = (b_memory[:, -N_STATES:]).reshape(-1, 224, 224, 4)[:, :, :, :3]
        dep_np_ = (b_memory[:, -N_STATES:]).reshape(-1, 224, 224, 4)[:, :, :, 3].reshape(-1, 224, 224, 1)
        dep_np_ = np.concatenate((dep_np_, dep_np_, dep_np_), axis=3)
        b_rgb_2 = torch.FloatTensor(rgb_np_).permute(0, 3, 1, 2).cuda()
        b_dep_2 = torch.FloatTensor(dep_np_).permute(0, 3, 1, 2).cuda()

        a = self.Actor_eval(b_rgb2, b_dep2, b_joint1)
        q = self.Critic_eval(b_rgb2, b_rgb2, b_joint1, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(b_rgb_2, b_dep_2, b_joint_1)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(b_rgb_2,b_dep_2, b_joint_1, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = b_r+GAMMA*q_  # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(b_rgb2,b_dep2, b_joint1, b_a)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

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
        s2 = np.array(s2).reshape(-1, 224 * 224 * 4)
        r = np.array(r).reshape(-1, 1)
        s3 = np.array(s3).reshape(-1, 3)
        s4 = np.array(s4).reshape(-1, 224 * 224 * 4)

        transition = np.hstack((s1, s2, a, r, s3, s4))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )
    def save_model(self):
        model_number = self.get_file_number("eval_ddpg")
        print("model_number:",model_number)
        torch.save(self.Actor_eval, "model/eval_ddpg/" + str(model_number) + ".pkl")
        torch.save(self.Critic_eval, "model/target_ddpg/" + str(model_number) + ".pkl")

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