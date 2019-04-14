from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.models as models

from collect_img_goal import read
from torch.utils.data import TensorDataset
import numpy as np
EPOCH = 100
BATCH_SIZE = 10
LR = 0.0005

print "-------{Loading data}-------"
x, y, data = read()
# print x[0], y[0], data[0]
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
x_y = torch.from_numpy(np.concatenate((x, y), axis=1))
data = torch.from_numpy(np.array(data)).permute(0, 3, 1, 2)

# print x_y.size(), data.size()

Train_DeepDataSet = TensorDataset(data[:10000], x_y[:10000])
Test_DeepDataSet = TensorDataset(data[21:230], x_y[21:230])
train_loader = Data.DataLoader(dataset=Train_DeepDataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_x = Variable(data[21:220].type(torch.FloatTensor)).cuda()  # todo
test_y = x_y[21:220].cuda()  # todo


# print test_x
print "-------{Load data finish!!!}-------"


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.dense121 = models.densenet121(True)
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


print "-------{Building Network}-------"
cnn = CNN().cuda()  # print cnn
#  cnn.cuda() todo
print "-------{Build finish!}-------"

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.MSELoss()


def cal_acc(a, b):
    sum1 = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            sum1 += 1
    return sum1


print "-------{Start Trian}-------"
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        print x.shape
        b_x = Variable(x).cuda()  # .cuda() todo
        b_y = Variable(y).cuda()  # .cuda() todo
        output = cnn(b_x).cpu().double()
        # print output, b_y
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # print test_x.shape
            test_out = cnn(test_x)
            pred_y = test_out.data
            # print pred_y[0], test_y[0]
            accuracy = float(cal_acc(test_y[0].numpy(), pred_y[0].numpy()))  # / test_y.size(0)
            print "Epoch:", epoch, "|train loss: %.4f" % loss.data.numpy(), "|test accuracy:%.4f" % accuracy

# test_out = cnn(test_x[:10])
#
# pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
# # pred_y = torch1.max(test_out, 1)[1].cuda().data.numpy().squeeze() todo
# # pred_y.cpu() todo
#
# print pred_y
# print test_y[:10].numpy()

