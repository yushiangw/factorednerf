import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pdb

import math


def act_factory(act, **kargs):

    if act=='leakyrelu':
        return lambda : nn.LeakyReLU(**kargs,inplace=True)
    elif act=='relu':
        return lambda : nn.ReLU(inplace=True) 
    elif act=='softplus':
        return lambda : nn.Softplus(**kargs) 
    else:
        raise Exception('error act type:'+act)



def make_mlp(in_dim, out_dim, hidden_dim, lnum, act_type, act_params, last_linear, use_wn ):

    make_act = act_factory(act_type, **act_params)

    layers=[]

    for i in range(lnum):
        
        isfirst = (i==0)
        islast  = (i==(lnum-1))

        if isfirst:
            in_ch = in_dim
            out_ch= hidden_dim
        elif islast:
            in_ch = hidden_dim
            out_ch= out_dim
        else:
            in_ch  = hidden_dim
            out_ch = hidden_dim

        mlp = nn.Linear(in_ch,out_ch)

        #-------------------------
        if act_type =='leakyrelu':
            torch.nn.init.kaiming_uniform_(
                        mlp.weight,
                        a=act_params['negative_slope'],
                        mode='fan_in',
                        nonlinearity='leaky_relu')


        #-------------------------
        if use_wn:
            mlp = nn.utils.weight_norm(mlp)
            

        #-------------------------
        if last_linear and islast:
            l = mlp
        else:
            act = make_act() 
            l = nn.Sequential(mlp,act)

        layers.append(l)


    model = nn.Sequential(*layers)

    return model 



"""
def make_geo_mlp(in_dim, out_dim, hidden_dim, lnum, act_type, act_params, last_linear, use_wn, radius_init=1.0, std=0.00001, ):


    make_act = act_factory(act_type, **act_params)
    layers=[]

    for i in range(lnum):
        isfirst = (i==0)
        islast  = (i==(lnum-1))

        if isfirst:
            in_ch = in_dim
            out_ch= hidden_dim
        elif islast:
            in_ch = hidden_dim
            out_ch= out_dim
        else:
            in_ch  = hidden_dim
            out_ch = hidden_dim

        #-------------------------

        mlp = nn.Linear(in_ch,out_ch)

        #-------------------------

        if islast:
            torch.nn.init.normal_(mlp.weight,
                        mean=math.sqrt(math.pi) / math.sqrt(in_ch),
                        std=std)

            torch.nn.init.constant_(mlp.bias, -radius_init)

        else:
            torch.nn.init.constant_(mlp.bias, 0.0) 
            torch.nn.init.normal_(mlp.weight, 0.0, 
                        math.sqrt(2.0) / math.sqrt(out_ch))

        #-------------------------
        if use_wn:
            mlp = nn.utils.weight_norm(mlp)

        if last_linear and islast:
            l = mlp
        else:
            l = nn.Sequential(mlp, make_act())

        layers.append(l)


    model = nn.Sequential(*layers)

    return model 


"""

class GeoSkipCMLP(nn.Module):

    def __init__(self, in_dim, coord_dim, out_dim, hidden_dim, lnum, act_type, act_params, skipc_layer_idx, last_linear, use_wn, radius_init, std):

        super().__init__() 
        #--------------------------------------------------

        self.skipc_layer_idx = skipc_layer_idx

        #--------------------------------------------------
        make_act = act_factory(act_type, **act_params)
        
        self.layers= nn.ModuleList()
        self.lnum = lnum 

        prev_o_chs=None 

        for i in range(lnum):
            isfirst = (i==0)
            islast  = (i==(lnum-1))

            if isfirst:
                in_ch = in_dim
                out_ch= hidden_dim

            elif islast:
                in_ch = hidden_dim
                out_ch= out_dim
            else:
                in_ch  = hidden_dim
                out_ch = hidden_dim

            #-------------------------
            # the layer before skip-connection
            if skipc_layer_idx != -1 and i==(skipc_layer_idx-1):
                out_ch= hidden_dim-in_dim
                assert out_ch>0
                prev_o_chs = out_ch

            #-------------------------

            mlp = nn.Linear(in_ch,out_ch)

            #-------------------------
            if isfirst:
                torch.nn.init.constant_(mlp.bias, 0.0) 
                torch.nn.init.normal_(mlp.weight, 0.0, 
                                    math.sqrt(2.0)/math.sqrt(out_ch))

                # FFN set to 0
                torch.nn.init.constant_(mlp.weight[:, coord_dim:], 0.0)

            elif i==skipc_layer_idx and skipc_layer_idx!=-1 :
                
                torch.nn.init.constant_(mlp.bias, 0.0) 
                torch.nn.init.normal_(mlp.weight, 0.0, 
                                      math.sqrt(2.0)/math.sqrt(out_ch))

                # [z, input]
                # FFN set to 0
                assert prev_o_chs is not None 
                torch.nn.init.constant_(
                            mlp.weight[:, (prev_o_chs+coord_dim):], 0.0)

            elif islast:
                torch.nn.init.normal_(mlp.weight,
                            mean=math.sqrt(math.pi) / math.sqrt(in_ch),
                            std=std)

                torch.nn.init.constant_(mlp.bias, -radius_init)

            else:
                torch.nn.init.constant_(mlp.bias, 0.0) 
                torch.nn.init.normal_(mlp.weight, 0.0, 
                            math.sqrt(2.0) / math.sqrt(out_ch))

            #-------------------------
            if use_wn:
                mlp = nn.utils.weight_norm(mlp)

            if last_linear and islast:
                l = mlp
            else:
                l = nn.Sequential(mlp, make_act())

            self.layers.append(l)


    def forward(self, x):

        x0 = x 
        z  = x

        for i in range(self.lnum):

            if self.skipc_layer_idx != -1 and i == self.skipc_layer_idx:
                # [z, input]
                z = torch.cat([z,x0],dim=-1)

            z = self.layers[i](z)


        return z 



class SkipCMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, lnum, act_type, act_params, skipc_layer_idx, last_linear, use_wn):

        super().__init__() 
        #--------------------------------------------------

        self.skipc_layer_idx = skipc_layer_idx

        #--------------------------------------------------
        make_act = act_factory(act_type, **act_params)
        
        in_full_dim = in_dim 

        self.layers= nn.ModuleList()
        self.lnum = lnum 

        for i in range(lnum):
            isfirst = (i==0)
            islast  = (i==(lnum-1))

            if isfirst:
                in_ch = in_full_dim
                out_ch= hidden_dim

            elif islast:
                in_ch = hidden_dim
                out_ch= out_dim

            else:
                in_ch  = hidden_dim
                out_ch = hidden_dim

            #-------------------------
            # the layer before skip-connection
            if skipc_layer_idx != -1 and i==(skipc_layer_idx-1): 
                out_ch= out_ch-in_full_dim
                assert out_ch>0

            #-------------------------

            mlp = nn.Linear(in_ch,out_ch)

            #-------------------------
            if use_wn:
                mlp = nn.utils.weight_norm(mlp)

            if last_linear and islast:
                l = mlp
            else:
                l = nn.Sequential(mlp, make_act())

            self.layers.append(l)


    def forward(self, x):

        x0 = x 
        z  = x

        for i in range(self.lnum):

            if self.skipc_layer_idx != -1 and i == self.skipc_layer_idx:
                # [z, input]
                z = torch.cat([z,x0],dim=-1)

            z = self.layers[i](z)


        return z 
