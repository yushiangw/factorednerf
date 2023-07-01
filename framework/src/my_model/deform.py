import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from functools import partial

import numpy as np
import pdb
import sys
import os 
import math

from einops import rearrange, reduce, repeat

from .blocks import make_mlp, SkipCMLP
from .freq   import WindowedPosEncoder
from .rigid_body import b_exp_se3
from .rigid_body_2d import b_simple_rot2d


class WarpField(nn.Module):

    def __init__(self, in_dim, use_pos_encoding, min_deg, max_deg,  z_dim ):
        super().__init__() 

        self.use_pos_encoding = use_pos_encoding
        self.min_deg = min_deg
        self.max_deg = max_deg

        if self.use_pos_encoding:

            assert self.max_deg-self.min_deg >0
            
            self.pos_encoder = WindowedPosEncoder(in_dim,
                                     min_deg, max_deg, 
                                     use_identity=True)

            x_dim = self.pos_encoder.get_output_dim()

        else:
            x_dim = in_dim

        self.backbone_mlp = SkipCMLP(
                                in_dim=x_dim+z_dim,  
                                out_dim=128,
                                hidden_dim=128,
                                lnum=6,
                                act_type='relu',
                                act_params={},
                                skipc_layer_idx=4,
                                last_linear=False,
                                use_wn=False ) 

        self.w_mlp    = nn.Linear(128,3)
        self.trsl_mlp = nn.Linear(128,3)

        nn.init.uniform_(self.w_mlp.weight,0,1.0e-4)
        nn.init.constant_(self.w_mlp.bias,0.0)

        nn.init.uniform_(self.trsl_mlp.weight,0,1.0e-4)
        nn.init.constant_(self.trsl_mlp.bias,0.0)

        # w = self.branches['w'](trunk_output)
        # v = self.branches['v'](trunk_output)
        # theta = jnp.linalg.norm(w, axis=-1)
        # w = w / theta[..., jnp.newaxis]
        # v = v / theta[..., jnp.newaxis]
        # screw_axis = jnp.concatenate([w, v], axis=-1)
        # transform  = rigid.exp_se3(screw_axis, theta)

    def forward(self, p, z, win_a=1.0):
        #
        # p: (b,3)
        #
        assert p.ndim ==2

        if self.use_pos_encoding:
            p2 = self.pos_encoder(p, alpha=win_a*self.max_deg)
        else:
            p2 = p 

        x = torch.cat([p2,z],dim=-1)

        x = self.backbone_mlp(x)

        # n,3
        w = self.w_mlp(x)
        v = self.trsl_mlp(x)

        # n,1
        theta = torch.norm(w,dim=-1) 
        theta_ = theta.unsqueeze(-1)

        w = w/ (theta_+1.0e-12)
        v = v/ (theta_+1.0e-12)

        # n,3,3
        R,trsl = b_exp_se3(  w=w, v=v, theta=theta )

        p2 = rearrange(p,'b c -> b c 1')

        q = torch.bmm(R,p2) + trsl 

        q = q.squeeze(-1)

        return q 



class BendMLP(nn.Module):
    # 
    # ambient slicing
    #
    # w_dim=2 
    # 
    def __init__(self, in_dim, use_pos_encoding, min_deg, max_deg, z_dim, w_dim):
        super().__init__()

        self.use_pos_encoding = use_pos_encoding
        self.min_deg = min_deg
        self.max_deg = max_deg
        
        if self.use_pos_encoding:
            self.pos_encoder = WindowedPosEncoder(in_dim,
                                     min_deg, max_deg, 
                                     use_identity=False)

            x_dim = self.pos_encoder.get_output_dim()

        else:
            x_dim = in_dim


        self.bend_mlp = SkipCMLP(
                        in_dim=x_dim+z_dim,  
                        out_dim=64,
                        hidden_dim=64,
                        lnum=5,
                        act_type='relu',
                        act_params={},
                        skipc_layer_idx=4,
                        last_linear=False,
                        use_wn=False )

        self.last_mlp = nn.Linear(64,w_dim)

        nn.init.normal_( self.last_mlp.weight, 0.0, 1.0e-5)
        nn.init.constant_(self.last_mlp.bias  ,0.0)


    def forward(self, p, z, win_a=1.0):
        #
        # p: (b,3)
        #
        assert p.ndim ==2

        if self.use_pos_encoding:
            p2 = self.pos_encoder(p, alpha=win_a*self.max_deg)
        else:
            p2 = p 

        x = torch.cat([p2,z],dim=-1)

        x = self.bend_mlp(x)
        w = self.last_mlp(x) 

        return w 


#==============================
class BijBlock(nn.Module): 

    def __init__(self, z_dim, 
                uv_min_deg, uv_max_deg, w_min_deg, w_max_deg ):

        super().__init__()

        hidden_dim = 128
        hf = hidden_dim//2
        z_out_dim = hf 

        #-----------------------------------
        self.uv_max_deg = uv_max_deg
        self.w_max_deg  = w_max_deg

        self.uv_pos_encoder = WindowedPosEncoder(
                                  2, 
                                 uv_min_deg, uv_max_deg, 
                                 use_identity=True)

        uv_pe_dim = self.uv_pos_encoder.get_output_dim()

        self.w_pos_encoder = WindowedPosEncoder(
                                 1, 
                                w_min_deg, w_max_deg, 
                                use_identity=True)

        w_pe_dim = self.w_pos_encoder.get_output_dim()

        #----------------------------------- 
        z_uv_out_dim = hidden_dim-uv_pe_dim
        assert z_uv_out_dim>0
        self.z_uv_mlp = nn.Sequential( 
                            nn.Linear(z_dim, z_uv_out_dim),
                            nn.ReLU(inplace=True))

        z_w_out_dim = hidden_dim-w_pe_dim
        assert z_w_out_dim>0
        self.z_w_mlp = nn.Sequential( 
                            nn.Linear(z_dim, z_w_out_dim),
                            nn.ReLU(inplace=True))

        #-----------------------------------
        self.uv_mlp = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),  
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, 1) )

        nn.init.uniform_(self.uv_mlp[-1].weight,0,1.0e-5)
        nn.init.constant_(self.uv_mlp[-1].bias,0.0)
        
        #-----------------------------------
        self.w_mlp = nn.Linear(hidden_dim, 3) 

        nn.init.uniform_(self.w_mlp.weight,0,1.0e-5)
        nn.init.constant_(self.w_mlp.bias,0.0)

        #-----------------------------------

    def forward(self, z, uv, w, win_a):
        assert  z.ndim==2
        assert  w.ndim==2
        assert uv.ndim==2

        B,C = z.shape  
        
        enc_uv = self.uv_pos_encoder(uv, alpha=win_a*self.uv_max_deg)

        z_uv = self.z_uv_mlp(z)

        # [code, enc_uv]
        x_uv = torch.cat([z_uv, enc_uv], dim=-1) 

        # mlp
        w_delta = self.uv_mlp(x_uv)

        pred_w = w + w_delta

        #-------------------------
        z_w  = self.z_w_mlp(z)

        enc_w  = self.w_pos_encoder(pred_w,
                     alpha=win_a*self.w_max_deg)

        # [code, enc_w]
        x_w = torch.cat([z_w, enc_w], dim=-1) 
        pred = self.w_mlp(x_w)

        theta = pred[:, 0]
        trsl  = pred[:,1:]

        pred_rot2d=b_simple_rot2d(theta)

        uv2 = rearrange(uv,'n c -> n c 1')
        trsl = rearrange(trsl,'n c -> n c 1')
        
        pred_uv = torch.bmm(pred_rot2d, uv2) + trsl
        pred_uv = pred_uv.squeeze(-1)

        return pred_uv, pred_w  

#==============================
class BijBlockWN(nn.Module): 

    def __init__(self, z_dim, 
                uv_min_deg, uv_max_deg, w_min_deg, w_max_deg, beta ):

        super().__init__()

        hidden_dim = 128
        hf = hidden_dim//2
        z_out_dim = hf 

        #-----------------------------------
        self.uv_max_deg = uv_max_deg
        self.w_max_deg  = w_max_deg

        self.uv_pos_encoder = WindowedPosEncoder(
                                  2, 
                                 uv_min_deg, uv_max_deg, 
                                 use_identity=True)

        uv_pe_dim = self.uv_pos_encoder.get_output_dim()

        self.w_pos_encoder = WindowedPosEncoder(
                                 1, 
                                w_min_deg, w_max_deg, 
                                use_identity=True)

        w_pe_dim = self.w_pos_encoder.get_output_dim()

        #----------------------------------- 
        wtnorm = nn.utils.weight_norm

        z_uv_out_dim = hidden_dim-uv_pe_dim
        assert z_uv_out_dim>0
        self.z_uv_mlp = nn.Sequential( 
                            wtnorm( nn.Linear(z_dim, z_uv_out_dim) ),
                            nn.Softplus(beta=beta))

        z_w_out_dim = hidden_dim-w_pe_dim
        assert z_w_out_dim>0
        self.z_w_mlp = nn.Sequential( 
                            wtnorm( nn.Linear(z_dim, z_w_out_dim) ),
                            nn.Softplus(beta=beta) )

        #-----------------------------------
        self.uv_mlp = nn.Sequential(
                            wtnorm( nn.Linear(hidden_dim, hidden_dim) ),  
                            nn.Softplus(beta=beta),
                            wtnorm( nn.Linear(hidden_dim,1)) ) 


        nn.init.uniform_(self.uv_mlp[-1].weight,0,1.0e-5)
        nn.init.constant_(self.uv_mlp[-1].bias,0.0)
        
        #-----------------------------------
        self.w_mlp = wtnorm(nn.Linear(hidden_dim, 3))

        nn.init.uniform_(self.w_mlp.weight,0,1.0e-5)
        nn.init.constant_(self.w_mlp.bias,0.0)

        #-----------------------------------

    def forward(self, z, uv, w, win_a):
        assert  z.ndim==2
        assert  w.ndim==2
        assert uv.ndim==2

        B,C = z.shape  
        
        enc_uv = self.uv_pos_encoder(uv, alpha=win_a*self.uv_max_deg)

        z_uv = self.z_uv_mlp(z)

        # [code, enc_uv]
        x_uv = torch.cat([z_uv, enc_uv], dim=-1) 

        # mlp
        w_delta = self.uv_mlp(x_uv)

        pred_w = w + w_delta

        #-------------------------
        z_w  = self.z_w_mlp(z)

        enc_w  = self.w_pos_encoder(pred_w,
                     alpha=win_a*self.w_max_deg)

        # [code, enc_w]
        x_w = torch.cat([z_w, enc_w], dim=-1) 
        pred = self.w_mlp(x_w)

        theta = pred[:, 0]
        trsl  = pred[:,1:]

        pred_rot2d=b_simple_rot2d(theta)

        uv2 = rearrange(uv,'n c -> n c 1')
        trsl = rearrange(trsl,'n c -> n c 1')
        
        pred_uv = torch.bmm(pred_rot2d, uv2) + trsl
        pred_uv = pred_uv.squeeze(-1)

        return pred_uv, pred_w  

#==============================
class BijMap(nn.Module): 

    def __init__(self, use_wn, *block_args, **block_kargs ):
        super().__init__()
    
        if use_wn:
            cl = partial(BijBlockWN, beta=100)
        else:
            cl = BijBlock

        self.maps = nn.ModuleList()
        for i in range(3):
            self.maps.append(cl(*block_args, **block_kargs )) 


    def forward(self, p, z, win_a): 

        assert p.ndim==2
        assert z.ndim==2

        q = p 

        idx_list=[
                [(0,1),(2)],
                [(0,2),(1)],
                [(1,2),(0)],
            ]

        undo_list=[
                (0,1,2),
                (0,2,1),
                (2,0,1),
            ]

        for i in range(len(self.maps)):

            uv_idx, w_idx=idx_list[i]

            uv=q[:,uv_idx]
            w =q[:,w_idx].unsqueeze(-1)

            i_map_block = self.maps[i]

            uv2,w2= i_map_block(z=z, uv=uv, w=w, win_a=win_a)

            uvw2 = torch.cat([uv2,w2],dim=-1)

            reset_idx = undo_list[i]

            q = uvw2[:, reset_idx].clone()


        return q 



