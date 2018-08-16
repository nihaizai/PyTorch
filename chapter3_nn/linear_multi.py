# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:09:19 2018

#多项式回归模型
@author: mmg
"""
import torch as th
import numpy as np
from torch.autograd import Variable


w_target = np.array([0.5,3,2.4])
b_target = np.array([0.9])

#目标函数 y = 0.9 + 0.5 * x + 3.0 * x^2 + 2.4 * x^3
x_sample = np.arange(-3,3.1,0.1)
y_sample = b_target[0] + w_target[0] * x_sample + w_target[1] * x_sample ** 2+\
           w_target[2] * x_sample ** 3
import matplotlib.pyplot as plt
#plt.plot(x_sample,y_sample,label='real curve')
#plt.legend()
#plt.show()           


x_train = np.stack([x_sample ** i for i in range(1,4)],axis = 1)
print('x_train： {}'.format(x_train))
x_train = th.from_numpy(x_train).float()

y_train = th.from_numpy(y_sample).float().unsqueeze(1)
print('y_train: {}'.format(y_train))

w = Variable(th.randn(3,1),requires_grad = True)
b = Variable(th.zeros(1),requires_grad = True)


x_train = Variable(x_train)
y_train = Variable(y_train)

def multi_linear(x):
    return th.mm(x,w) + b  #torch.mm 矩阵乘法

def get_loss(y,y_):
    return th.mean((y-y_)**2)

#y_pred = multi_linear(x_train)
#
#plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label = 'fitting curve',color='r')
#plt.plot(x_train.data.numpy()[:,0],y_sample,label = 'real curve',color='b')
#plt.legend()
#plt.show()
#
#
#loss = get_loss(y_pred,y_train)
#
#loss.backward()
#print(w.grad)
#print(b.grad.data)

ln = 0.001
#w.data = w.data - ln*w.grad.data
#b.data = b.data - ln*b.grad.data
#y_pred = multi_linear(x_train)
#
#plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label = 'fitting curve',color='r')
#plt.plot(x_train.data.numpy()[:,0],y_sample,label = 'real curve',color='b')
#plt.legend()
#plt.show()

epochs = 100
for e in range(epochs):
    y_pred = multi_linear(x_train)
    loss = get_loss(y_pred,y_train)
    

    loss.backward()
    
    w.data = w.data - ln * w.grad.data
    b.data = b.data - ln * b.grad.data
    
    w.grad.data.zero_()     #当只写epochs这个循环时，w.grad.data.zero_()应放在更新后面，为什么？？、
    b.grad.data.zero_()
    
    if (e+1)%20 == 0:
        print('epochs:{},loss:{:.5f}'.format(e+1,loss.data))


print('w.data:{}'.format(w.data))
print('b.data:{}'.format(b.data))
y_pred = multi_linear(x_train)


plt.plot(x_train.data.numpy()[:,0],y_pred.data.numpy(),label = 'fitting curve',color='r')
plt.plot(x_train.data.numpy()[:,0],y_sample,label = 'real curve',color='b')
plt.legend()
plt.show()
      


    