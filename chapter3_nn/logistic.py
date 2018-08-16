# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:03:17 2018

#Logistic 回归模型
@author: mmg
"""
import torch
import torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt 

torch.manual_seed(2017)
with open('./data.txt','r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]


#标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max,i[1]/x1_max) for i in data]

x0 = list(filter(lambda x:x[-1] == 0.0,data))
x1 = list(filter(lambda x:x[-1] == 1.0,data))

#画出数据点
plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

plt.plot(plot_x0,plot_y0,'ro',label='x_0')
plt.plot(plot_x1,plot_y1,'bo',label='x_1')
plt.legend(loc='best')
plt.show()

np_data = np.array(data,dtype='float32')
x_data = torch.from_numpy(np_data[:,0:2])
y_data = torch.from_numpy(np_data[:,-1]).unsqueeze(1)

#sigmoid函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#sigmoid图像
plot_x = np.arange(-10,10.01,0.01)
plot_y = sigmoid(plot_x)
plt.plot(plot_x,plot_y,'r')

x_data = Variable(x_data)
y_data = Variable(y_data)

import torch.nn.functional as F
# 定义logistic 回归模型
w = Variable(torch.randn(2,1),requires_grad = True)
b = Variable(torch.zeros(1),requires_grad = True)

def logistic_regression(x):
    return F.sigmoid(torch.mm(w,x) + b) #PyTorch中已经封装好了sigmoid函数

w0 = w[0].data[0]
w1 = w[1].data[1]
b0 = b.data[0]
print('b0: {}'.format(b0))

plot_x = np.arange(0.2,1,0.01)
plot_y = (-w0 * plot_x - b0) /w1  #?????

plt.plot(plot_x,plot_y,'g',label='cutting line')
plt.plot(plot_x0,plot_y0,'ro',label='x_0')
plt.plot(plot_x1,plot_y1,'bo',label='x_1')
plt.legend()
plt.show()

#loss
def binary_loss(y_pred,y):
    logits = (y * y_pred.clamp(1e-12).log() + (1-y)*(1-y_pred).clamp(1e-12).log()).mean()
    return -logits

y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred,y_data)
print loss

loss.backward()
w.data = w.data - 0.1 * w.grad.data
b.data = b.data - 0.1 * b.grad.data

y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred,y_data)
print loss









#g = lambda x: x*x-4
#for i in range(10):
#    print g(i)
#print filter(lambda x:x*x-4,range(10))


