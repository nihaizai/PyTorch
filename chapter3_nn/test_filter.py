# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:31:54 2018

@author: mmg
#lambda filter map reduce
"""



#print [(lambda x : x*x - 4)(x) for x in range(4)]
#
#print (lambda x:x*x - 4)(3)
#
#g = lambda x: x*x-4
#for i in range(10):
#    print g(i)
#print filter(lambda x:x*x-4,range(10))
#print map(lambda x:x*x-4,range(10))
#
#print map(lambda x,y:x*y-4,range(3),[8,9,10])

print reduce(lambda x,y:x*y-4,range(4))

print reduce(lambda x,y:x+y,range(101))

print reduce(lambda x,y:x+y,range(101),100)