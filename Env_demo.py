#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np

# 超参数
CLOSED_POS = 0.0  # The position for a fully-closed gripper (meters).
OPENED_POS = 0.10  # The position for a fully-open gripper (meters).
ACTION_SERVER = 'gripper_controller/gripper_action'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Robot(object):
    MIN_EFFORT = 35  # Min grasp force, in Newtons
    MAX_EFFORT = 100  # Max grasp force, in Newtons
    dt = 0.005  # 转动的速度和 dt 有关
    action_bound = [-1, 1]  # 转动的角度范围
    state_dim = 3  # 7个观测值
    action_dim = 3  # 7个动作
    Robot = {'fetch': {'init': [0, 0, 0]}}

    def __init__(self):
        self.cont = 0
        self.dis = 100
        self.Box_position = [0.6, 0.1, 0.65]
        self.arm_goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_goal = [0.0, 0.0, 0.0]
        # 初始化reward
        self.dis = 10
        self.reward = math.exp(-self.dis)
        self.RGBDimg = np.random.randn(224, 224, 4)
        # 更新场景到rviz中

    def get_state(self):
        return self.end_goal, self.RGBDimg

    def test_step(self, action, var):
        # self.cont+=1
        done = False
        self.end_goal += action[0]
        # print "---frist---"
        self.end_goal[0] = np.random.normal(self.end_goal[0], var)%0.4 + 0.52
        self.end_goal[1] = np.random.normal(self.end_goal[0], var)%0.7 - 0.35
        # x = self.end_goal[0]
        # y = self.end_goal[1]
        x = self.end_goal[0]
        y = self.end_goal[1]
        # print(x, y)
        dis = math.sqrt(math.pow(x - self.Box_position[0], 2)
                        + math.pow(y - self.Box_position[1], 2))
        if dis < 0.02:  # 阈值，可调
            done = True
            reward = 100
        else:
            reward = -dis
            reward *= 10
        self.dis = dis
        # if(self.cont>5000):
        #     print(x,y, self.Box_position, dis, reward)
        new_position = [x, y, 1.0]
        return new_position, reward, done
