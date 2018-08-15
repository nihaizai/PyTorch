# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:57:29 2018
#对stack进行测试
@author: mmg
"""

import numpy as np

#a = [[1,2,3],[4,5,6]]
#print('a:{}'.format(a))
#
#c = np.stack(a,axis = 0)
#print('c:{}'.format(c))
#
#d = np.stack(a,axis = 1)
#print('d:{}'.format(d))


array = [[[1,2,3,4],[1,2,3,4],[1,2,3,4]]]
print array

arrays = np.array(array)
print arrays
#print arrays[0]
#print arrays[0][0]
#print arrays[0][0][0]
#
#
print np.stack(arrays,axis = 0)
print np.stack(arrays,axis = 1) 
print np.stack(arrays,axis = 2)


a = np.array([[1,2,3,4],[5,6,7,8]])
arrays2 = np.asarray([a,a,a])
#print arrays2

#print arrays2[0]
#print arrays2[0][0]
#print arrays2[0][0][0]
print np.stack(arrays2,axis = 0)
print np.stack(arrays2,axis = 1)
print np.stack(arrays2,axis = 2)

