import sys
sys.path.append('..')

import torch

def simple_batch_norm_1d(x,gamma,beta):
    eps = 1e-5
    x_mean = torch.mean(x,dim=0,keepdim=True)
    x_var = torch.mean((x - x_mean) ** 2,dim=0,keepdim=True)
    x_hat = (x - x_mean)/ torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

x = torch.arange(15).view(5,3)
gamma = torch.ones(x.shape[1])
beta = torch.zeros(x.shape[1])
print('before bn: ')
print(x)

y = simple_batch_norm_1d(x,gamma,beta)
print('agter bn: ')
print(y)


    
    
