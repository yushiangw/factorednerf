import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat 


from ..rotation_conversions import quaternion_to_matrix, matrix_to_quaternion


def quat2mat44(quat,trsl):
    assert quat.ndim==2 and quat.shape[1]==4
    assert trsl.ndim==2 and trsl.shape[1]==3

    # b,3,3
    rot = quaternion_to_matrix(quat)
    
    # b,3
    trsl = rearrange(trsl,'b c -> b 1 c')

    B  = quat.shape[0]
    RT = torch.eye(4,device=quat.device)
    RT = repeat(RT,' r w -> b r w', b=B).detach()

    RT[:,:3,:3]=rot
    RT[:,:3, 3]=trsl

    return RT 


def apply_SE3_transform(p, RT, apply_trsl):
    assert p.ndim==3
    assert RT.ndim==3

    # b 4 4 
    # RT 

    # b c n 
    _p = rearrange(p, 'b n c -> b c n')

    # b c n 
    #q = torch.matmul(RT[:,:3,:3], _p)
    q = torch.bmm(RT[:,:3,:3], _p)
    
    if apply_trsl:
        _t = rearrange(RT[:,:3,3], 'b c -> b c 1')
        q = q + _t  

    
    q = rearrange(q, 'b c n -> b n c')

    return q


class SE3_OptBlock(nn.Module):

    def __init__(self, use_parameter):

        super().__init__()  

        self.use_parameter=use_parameter

        if use_parameter:
            self.pose = nn.Parameter(torch.zeros(1,7))
            self.pose.data[0,0]=1.0 
        else:
            self.register_buffer('pose', torch.zeros(1,7))
            self.pose[0,0]=1.0 

        self.log_grad_module=[]

    def reset(self, quat, trsl): 

        if self.use_parameter:
            self.pose.data[:,:4] = quat.clone().detach()
            self.pose.data[:,4:] = trsl.clone().detach()
        else:
            self.pose[:,:4] = quat.clone().detach()
            self.pose[:,4:] = trsl.clone().detach()

    def decom(self ): 
        # 1 7
        quat = self.pose[:,:4]
        trsl = self.pose[:,4:]
        return quat,trsl

    def to_RT(self ): 
        # 1 7
        quat,trsl = self.decom()
        # quat = self.pose[:,:4]
        # trsl = self.pose[:,4:]

        # 1,4,4 
        RT = quat2mat44(quat,trsl) 
        return RT,quat,trsl

    def apply(self, p, apply_trsl):
        
        assert p.ndim==2 

        RT,quat,trsl = self.to_RT()  

        _p = p.unsqueeze(0)
        q = apply_SE3_transform(_p, RT, apply_trsl)
        q = q.squeeze(0)

        return q

    def apply_inv(self, p, apply_trsl):
        
        assert p.ndim==2 

        RT,quat,trsl = self.to_RT()  
        
        # invRT = torch.zeros_like(RT)
        # invRT[:, 3, 3]= 1.0
        # invRT[:,:3,:3]= RT[:, :3,:3].permute(0,2,1)
        # invRT[:,:3 ,3]= RT[:, :3, 3]*-1
        invRT = torch.inverse(RT)

        _p = p.unsqueeze(0)
        q = apply_SE3_transform(_p, invRT, apply_trsl)
        q = q.squeeze()

        return q 

    def forward(self,):
        # 
        # p: b n 3
        # z: b n z 

        RT,quat,trsl = self.to_RT()  

        # rot = RT[:,:3,:3] # 1,3,3
        # trsl= RT[:,:3, 3] # 1,3 
        invRT = torch.inverse(RT)
        # invRT = torch.zeros_like(RT)
        # invRT[:, 3, 3]= 1.0
        # invRT[:,:3,:3]= RT[:, :3,:3].permute(0,2,1)
        # invRT[:,:3 ,3]= RT[:, :3, 3]*-1 
        out={
            'RT':RT,
            # 'rot':rot,
            # 'trsl':trsl,
            'invRT':invRT,
            'quat':quat,
        }

        return out

