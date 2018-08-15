# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:24:14 2018
#对unsqueeze()进行测试
@author: mmg
"""

import torch

#squeeze()
print('a:    ')
a = torch.Tensor(1,3)
print a

print a.squeeze(0)

print a.squeeze(1)

print('b:    ')
b = torch.Tensor(2,3)
print b

print b.squeeze(0)

print b.squeeze(1)

print('c:    ')
c = torch.Tensor(3,1)
print c
print c.squeeze(0)
print c.squeeze(1)

print('d:    ')
d = torch.rand(4,1,3)
print d
print d.squeeze()

print('e:    ')
e = torch.rand(4,3,1)
print e
print e.squeeze()


print('f:    ')
f = torch.rand(4,3,2)
print f
print f.squeeze()



#unsqueeze()
print ('g:  ')
g = torch.Tensor(3)
print g
print g.unsqueeze(0)

print g.unsqueeze(1)


#print g.unsqueeze() 必须指明维度

