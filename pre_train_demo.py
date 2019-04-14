#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
from src.fetch_moveit_config.DQN import DQN
from src.fetch_moveit_config.DDPG import DDPG
from src.fetch_moveit_config.Env_demo import Robot
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
MAX_EPISODES = 10000
MAX_EP_STEPS = 20
MEMORY_CAPACITY = 5000


def rand_read_img():
    np_data_number = []
    for root, dirs, files in os.walk("/home/ljt/images/rgb/"):
        np_data_number = files
    temp = np.random.randint(1, len(np_data_number))
    position = np_data_number[temp][:-4][1:-1].split(",")
    img_sample = cv2.imread("/home/ljt/images/rgb/"+np_data_number[temp])
    dep_sample = np.load("/home/ljt/images/dep/"+np_data_number[temp][:-4]+".npy")
    data_sample = np.concatenate((img_sample, dep_sample.reshape(224, 224, -1)), 2)

    for item in range(len(position)):
        position[item] = float(position[item])
    return position, data_sample


def file_name(file_dir="/home/ljt/ws/src/fetch_moveit_config/data/memory"):
    for root, dirs, files in os.walk(file_dir):
        for item in range(len(files)):
            files[item] = int(files[item][:-4])
        return max(files)


def load_model():
    # print "----loading previous model----"
    model_number = rl.get_file_number("eval_dqn")
    while model_number > 0:
        try:
            model_number = rl.get_last_model("eval_dqn")
            rl.eval_net = torch.load("model/eval_dqn/" + str(model_number - 1) + ".pkl").cuda()
            rl.target_net = torch.load("model/target_dqn/" + str(model_number - 1) + ".pkl").cuda()
            break
        except Exception as e:
            model_number -= 1
            continue


if __name__ == "__main__":
    robot = Robot()
    s_dim = robot.state_dim
    a_dim = robot.action_dim
    a_bound = robot.action_bound
    var = 3
    rl = DDPG()

    # load_model()

    recent_end_goal = [0.127498550885, 0.370453284224, 1.16039135584]  # 初始的夹爪位置
    begin = time.clock()
    # print "loading memory!!!"
    # rl.memory = np.load("/home/ljt/ws/src/fetch_moveit_config/data/memory/"+str(file_name())+".npy")
    # 获取物块位置
    Box_position, view_img = rand_read_img()  # new we wet only one position
    robot.RGBDimg = view_img
    robot.Box_position = Box_position
    print(Box_position)
    # plt.imshow(view_img[:, :, :3]/255)
    # plt.show()

    for i in range(1, MAX_EPISODES):
        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
        st = 0
        rw = 0
        # print "cube position:", Box_position
        # 存储夹爪距离木块的距离
        now_dis = math.sqrt(math.pow(recent_end_goal[0] - Box_position[0], 2)
                            + math.pow(recent_end_goal[1] - Box_position[1], 2))
        robot.dis = now_dis
        # 存储end_goal
        robot.end_goal = recent_end_goal
        observation = robot.get_state()
        if i % 500 == 0:
            print("****************memory counter:{0}****************".format(rl.memory_counter))
        end = time.clock()
        print(end-begin)
        begin = time.clock()
        # 分成末端坐标和rgbd
        endg, view_state = observation
        while True:
            st += 1
            action = rl.choose_action([endg, view_state])  # choose 时沿用之前的图像
            recent_end_goal = action[0]
            observation_, r, done = robot.test_step(action, var)  # 执行一步
            rl.store_transition(observation, action, r, [observation_, view_state])  # 沿用之前的图像 RGBD
            if rl.memory_counter > 5000:  # and st % 2 == 0:
                var *= .9995
                rl.learn()
                if st == 1:
                    print(".....................learn.....................")
                    # if i % 10 == 0:
                    #     rl.save_model()
            rw += r
            endg = observation_
            if done or st >= MAX_EP_STEPS:
                print("Step:{0}, total reward:{1}, average reward:{2}\n".format(st, rw, rw*1.0/st))
                break
            observation = [observation_, view_state]
        if i % 500 == 0:
            print("saving memory!!!")
            # np.save("/home/ljt/ws/src/fetch_moveit_config/data/memory/"+str(rl.memory_counter), rl.memory)
            # print "sucess saved!!!"
