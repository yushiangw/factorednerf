
import torch
import numpy as np
from einops import rearrange, reduce, repeat 




def b_simple_rot2d(theta):

    assert theta.ndim==1

    B = theta.shape[0]

    rot2  = torch.ones((2,2),device=theta.device,dtype=theta.dtype)
    brot2 = repeat(rot2, 'r c -> b r c ', b=B).clone().detach()

    sin_theta = torch.sin(theta)  
    cos_theta = torch.cos(theta)

    brot2[:,0,0]=cos_theta.clone()
    brot2[:,0,1]=-1*sin_theta.clone()
    brot2[:,1,0]=sin_theta.clone()
    brot2[:,1,1]=cos_theta.clone()

    return brot2 

