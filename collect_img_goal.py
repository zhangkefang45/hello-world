import math
from camera import RGBD
from MofanDDPG import DDPG
from Env import Robot, CubesManager
import copy
import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import cv2

MAX_PICTURE_NUM = 30000


def read(file_dir="/home/ljt/Desktop/data"):
    x = []
    y = []
    data = []
    number = 0
    for root, dirs, files in os.walk(file_dir):
        for i in files:
            # print "the file data:", np.load("/home/ljt/Desktop/data/"+i)
            x_y = i[1:-11].split(" ")
            x1 = float(x_y[0][:-1])
            y1 = float(x_y[1])
            data1 = np.load("/home/ljt/Desktop/data/"+i)
            x.append(x1)
            y.append(y1)
            number += 1
            data.append(data1)
            if number == 1:
                break
            # print x1, y1, data1
    return x, y, data


def collect_data():
    robot = Robot()
    cubusm = CubesManager()
    for i in range(MAX_PICTURE_NUM):
        cubusm.reset_cube(rand=True)
        Box_position = cubusm.read_cube_pose("demo_cube")
        # print "cube position:", str(Box_position)
        joint, view = robot.get_state()
        rgb, dep = robot.get_rgb_dep()
        # b, g, r = cv2.split(rgb)
        # print view[0,0,0]
        # print dep
        # rgb = cv2.merge([r, g, b])
        # print dep
        # plt.imshow(dep)
        # plt.show()
        rgb = cv2.resize(rgb,(224,224))
        dep = cv2.resize(dep,(224,224))
        # print dep
        cv2.imwrite("/home/ljt/Desktop/images/rgb/" + str(Box_position) + ".png", rgb)
        # cv2.imwrite("/home/ljt/Desktop/ws/src/fetch_moveit_config/images/dep/" + str(Box_position) + ".png", dep)
        # a = np.array(rgb).shape
        # print a
        # print "camera image shape:", view.shape
        np.save("/home/ljt/Desktop/images/dep/"+str(Box_position), dep)


def read_image_():
    for root, dirs, files in os.walk("/home/ljt/Desktop/images/rgb/"):
        for i in range(30000):
            img_sample = cv2.imread("/home/ljt/Desktop/images/rgb/"+files[i])
            dep_sample = np.load("/home/ljt/Desktop/images/dep/"+files[i][:-4]+".npy")
            print img_sample
            print dep_sample.reshape(224, 224, -1)
            data_sample = np.concatenate((img_sample, dep_sample.reshape(224, 224, -1)), 2)
            break


if __name__ == '__main__':
    # collect_data()
    read_image_()
    # a = cv2.imread("/home/ljt/Desktop/ws/src/fetch_moveit_config/images/rgb/[0.8944105684170961, 0.33323058418776674, 0.75].png")
    # cv2.imshow("name", a)
    # print a.shape
    # # x, y, data = read("/home/ljt/Desktop/data")
    # # print data[0][:, :, 3]
    # # print data[0][0, 0, :]
    # # print data[0][:, :, :3]
    # a = []
    # for root, dirs, files in os.walk("/home/ljt/Desktop/data"):
    #     a = files
    # temp = np.random.randint(1, len(a))
    # print np.load("/home/ljt/Desktop/data/"+a[temp])
    # position = a[temp][:-4][1:-1].split(",")
    # for i in range(len(position)):
    #     position[i] = float(position[i])
    # print (position)
