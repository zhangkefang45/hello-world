#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
from camera import RGBD
from DDPG import  DDPG
from DQN import DQN
from Env import Robot, CubesManager
import time
import copy
import rospy
import numpy as np
MAX_EPISODES = 10000
MAX_EP_STEPS = 7
MEMORY_CAPACITY = 10000



def rand_read_img(file_dir="/home/ljt/Desktop/data"):
    a = []
    for root, dirs, files in os.walk(file_dir):
        a = files
    temp = np.random.randint(1, len(a))
    position = a[temp][:-4][1:-1].split(",")
    for i in range(len(position)):
        position[i] = float(position[i])
    return position, np.load("/home/ljt/Desktop/data/"+a[temp])


def file_name(file_dir="/home/ljt/Desktop/ws/src/fetch_moveit_config/data/memory"):
    for root, dirs, files in os.walk(file_dir):
        for i in range(len(files)):
            files[i] = int(files[i][:-4])
        return max(files)


if __name__ == "__main__":
    robot = Robot()
    s_dim = robot.state_dim
    a_dim = robot.action_dim
    a_bound = robot.action_bound
    cubm = CubesManager()
    rl = DQN()
    print "----loading previous model----"
    rl.eval_net = torch.load('eval_dqn.pkl').cuda()
    rl.target_net = torch.load('target_dqn.pkl').cuda()

    robot.reset()
    start_position = robot.gripper.get_current_pose("gripper_link").pose.position  # 初始的夹爪位置
    begin = time.clock()
    # print "loading memory!!!"
    # rl.memory = np.load("/home/ljt/Desktop/ws/src/fetch_moveit_config/data/memory/"+str(file_name())+".npy")
    for i in range(1, MAX_EPISODES):
        print "\n------Episode:{0}------".format(i)
        cubm.reset_cube(rand=True)
        Box_position = cubm.read_cube_pose("cube1")
        st = 0
        rw = 0
        # 获取物块位置
        # print "cube position:", Box_position
        robot.Box_position = copy.deepcopy(Box_position)
        now_dis = math.sqrt(math.pow(start_position.x - robot.Box_position[0], 2)
                            + math.pow(start_position.y - robot.Box_position[1], 2))
        # 存储夹爪距离木块的距离

        robot.dis = now_dis  # + math.pow(now_position.z - robot.Box_position[2], 2))
        # 存储end_goal
        robot.end_goal = [start_position.x, start_position.y, start_position.z]
        s = robot.get_state()
        if i % 500 == 0:
            print "********memory counter:{0}********".format(rl.memory_counter)
        end = time.clock()
        print end-begin
        begin = time.clock()
        # 分成末端坐标和rgbd
        endg, view_state = s
        rgb, dep = robot.get_rgb_dep()
        for j in range(1, MAX_EP_STEPS):
            st += 1
            a = rl.choose_action([endg, view_state])               # choose 时沿用之前的图像
            s_, r, done = robot.test_step(a)                       # 执行一步
            rl.store_transition(s, a, -r, [s_, view_state])       # 沿用之前的图像rgbd
            # print "the memory counter:", rl.memory_counter
            if rl.memory_counter > 5000:
                # if rl.memory_counter % 500 == 0:
                print "learn....."
                rl.learn()
            rw += r
            if done or st >= MAX_EP_STEPS-1:
                print("total reward:{0}, average reward:{1}\n".format(rw, rw*1.0/st))
                break
        if i % 500 == 0:
            print "saving memory!!!"
            # np.save("/home/ljt/Desktop/ws/src/fetch_moveit_config/data/memory/"+str(rl.memory_counter), rl.memory)
            # print "sucess saved!!!"
