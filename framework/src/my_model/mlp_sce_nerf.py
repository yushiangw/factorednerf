import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import pdb
import sys
import os 
import math
from einops import rearrange, reduce, repeat

from .blocks import make_mlp, GeoSkipCMLP
from .freq  import PosEncoder,  WindowedPosEncoder
from . import grad 


class SDF_MLP(nn.Module):

    def __init__(self, input_dim,
                       input_use_encoding,
                       input_min_deg,
                       input_max_deg,
                       sdf_lnum,
                       sdf_hdim, 
                       out_rgb_hdim,
                       use_geo_ini,  
                       use_wn,
                       t_dim, 
                       append_t, 
                       init_variance,
                       max_invs,
                       use_variance,
                       geo_radius_init,
                       geo_std,
                       geo_skipc_layer_idx,
                       use_bending,
                       bend_wdim,
                    ):

        super().__init__() 

        # self.min_bound = min_bound
        # self.grid_size = grid_size
        self.input_use_encoding = input_use_encoding
        self.append_t = append_t    
        self.use_bending =use_bending
        self.max_invs=max_invs
        self.use_variance=use_variance
        
        if self.input_use_encoding:   

            assert input_max_deg-input_min_deg >0

            self.input_min_deg = input_min_deg
            self.input_max_deg = input_max_deg

            self.pos_encoder = WindowedPosEncoder(input_dim,
                                                  input_min_deg, 
                                                  input_max_deg, 
                                                  use_identity=True)

            enc_p_dim = self.pos_encoder.get_output_dim()

        else:
            enc_p_dim = input_dim 


        ex_dim=0
        if self.append_t:
            ex_dim += t_dim 

        if self.use_bending:
            ex_dim += bend_wdim

        #-------------------------------------------
        
        sdf_in_ft_dim = enc_p_dim + ex_dim

        if use_geo_ini:

            self.decoder =  GeoSkipCMLP(
                                in_dim=sdf_in_ft_dim,
                                coord_dim=3,
                                out_dim=1+out_rgb_hdim,
                                hidden_dim=sdf_hdim,
                                lnum=sdf_lnum,
                                act_type='softplus', 
                                act_params={'beta':100}, 
                                skipc_layer_idx=geo_skipc_layer_idx,
                                last_linear=True, 
                                use_wn=use_wn,
                                radius_init=geo_radius_init,
                                std=geo_std ) 
        

        else:
            self.decoder = make_mlp( 
                                in_dim=sdf_in_ft_dim,
                                out_dim=1+out_rgb_hdim,
                                hidden_dim=sdf_hdim,
                                lnum=sdf_lnum,
                                act_type='leakyrelu', 
                                act_params={'negative_slope':0.01}, 
                                last_linear=True, 
                                use_wn=use_wn ) 
            
        #---------------------------
        # Variance for convert from sdf to density
        if self.use_variance:
            self.variance: nn.Parameter = nn.Parameter(
                            torch.tensor(init_variance))
        else:
            self.register_buffer('variance', torch.ones((1)))
        #---------------------------
        # self.log_grad_plot_module=[]
        self.log_grad_module=[self.decoder]  

    def __get_var(self):
        return self.variance
        
    def __get_invs(self):
        return torch.exp(self.variance*10).clip(1e-6, self.max_invs)

    def forward(self, tx_noc_p, win_a, t_code=None, amb_w=None):

        if self.input_use_encoding:
            enc_p = self.pos_encoder(tx_noc_p,  
                        alpha=win_a*self.input_max_deg)
        else:
            enc_p = tx_noc_p 

        #--------------------
        # n,c
        sdf_ft = enc_p

        if self.append_t :
            # n,t
            #t_code = motion_net.get_time_code(p_tidx)
            assert t_code is not None
            # n,c*k+t
            sdf_ft = torch.cat([sdf_ft,t_code],dim=-1) 

        if self.use_bending:
            assert amb_w is not None
            sdf_ft = torch.cat([sdf_ft,amb_w],dim=-1) 

        #--------------------
        _y = self.decoder(sdf_ft)

        _p_sdf = _y[:,:1]
        _p_h   = _y[:,1:]

        invs= self.__get_invs()
        invs = invs.expand(_p_sdf.shape)

        return _p_sdf, _p_h, invs


#-----------------------------------------------------------
class NeuSTrackDistll_MainMLP(nn.Module):

    def __init__(self,
                 rgb_in_ft_dim,
                 rgb_lnum,
                 rgb_hdim,
                 rgb_use_viewdir, 
                 rgb_use_nv,
                 rgb_use_wn,
                 skip_outter_samples,
                 append_t,
                 use_bending,
                 bend_wdim,
                 trackft_lnum,
                 trackft_hdim,
                 trackft_act,
            ):

        super().__init__() 

        self.rgb_use_viewdir = rgb_use_viewdir
        self.rgb_use_nv = rgb_use_nv 

        self.append_t=append_t
        self.use_bending=use_bending

        self._deform_win_a=1.0
        self._deform_win_b=1.0
        
        self.skip_outter_samples = skip_outter_samples
        self.bend_wdim=bend_wdim # for rigid object awb=0

        #-------------------------------------------   
        # [ft,p,nv,view]
        # encodeing 
        #   ft: h
        #   p:  x
        #   nv: 3
        #   view: 3+2*4*3

        rgb_in_dim = rgb_in_ft_dim 
        tk_in_dim  = rgb_in_ft_dim

        if self.rgb_use_nv:
            rgb_in_dim +=3

        if self.use_bending:
            rgb_in_dim += bend_wdim

            _amb_w = torch.zeros((bend_wdim),dtype=torch.float32)
            nn.init.uniform_(_amb_w,0.0,1.0e-4)
            
            self.register_buffer('rigid_amb_w', _amb_w )

        if self.rgb_use_viewdir:

            # 3*4*2=24
            self.rgb_viewdir_encoder = WindowedPosEncoder(
                                            3,0,4, 
                                            use_identity=True)

            rgb_in_dim +=  self.rgb_viewdir_encoder.get_output_dim()

        
        self.rgb_decoder =make_mlp(
                                in_dim=rgb_in_dim, 
                                out_dim=3,
                                hidden_dim=rgb_hdim,
                                lnum=rgb_lnum,
                                act_type='relu', 
                                act_params={},
                                last_linear=True,
                                use_wn=True ) 


        if trackft_act=='relu':
            act_type='relu'
            act_params={}
        elif trackft_act=='leakyrelu':
            act_type='leakyrelu'
            act_params={'negative_slope':0.01}
        else:
            raise Exception('unknown trackft_act:'+trackft_act)

        self.tkft_decoder =make_mlp(
                                in_dim=tk_in_dim, 
                                out_dim=256,
                                hidden_dim=trackft_hdim,
                                lnum=trackft_lnum,
                                act_type=act_type,
                                act_params=act_params,
                                last_linear=True,
                                use_wn=True ) 

        #---------------------------
        self.log_grad_module=[self.rgb_decoder,self.tkft_decoder]  


    def convert_coord(self, p, min_bound, grid_size):  
        assert p.ndim==2
        assert min_bound.ndim==2
        assert grid_size.ndim==2

        max_v = grid_size.max()
        # noc_p = ((p - min_bound)/grid_size)*2-1.0
        noc_p = ((p - min_bound)/max_v)*2-1.0
        return noc_p 

    def forward(self, p, vdirs, scene_data, normalize, motion_net, p_tidx, deform=False, sdf_only=False):
        
        assert p.ndim==2 
        if vdirs is not None:
            assert p.shape==vdirs.shape, f'{p.shape} {vdirs.shape}'

        all_ft=[]
        all_valid=[]

        sdf_decoder = scene_data['model']
        grid_size = scene_data['grid_size']
        min_bound = scene_data['min_bound']
        
        # grid_size = sdf_decoder.grid_size
        # min_bound = sdf_decoder.min_bound

        #--------------------------------------------
        # normalize points
        if normalize:
            noc_p = self.convert_coord(p, min_bound, grid_size)
        else:
            noc_p = p 

        #--------------------------------------------
        # deformation block

        if deform:
            assert motion_net is not None
            assert p_tidx is not None
            assert p_tidx.shape[0] == noc_p.shape[0]

            if self.use_bending:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a,
                                    win_b=self._deform_win_b,)
                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = rt['w']

            else:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a )

                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = None

        else:
            p_shift = None 
            tx_noc_p= noc_p

            if self.use_bending:

                amb_w = repeat(self.rigid_amb_w, 
                              'c -> n c', n=noc_p.shape[0]).clone()

                #amb_w = torch.zeros(
                #            (noc_p.shape[0], self.bend_wdim),
                #            device=noc_p.device)
            else:
                amb_w = None

        #--------------------
        # n,c 
        if self.append_t :
            # n,t
            t_code = motion_net.get_time_code(p_tidx)
        else:
            t_code = None

        #--------------------------------------------        
        if tx_noc_p.requires_grad ==False:
            tx_noc_p.requires_grad=True
        
        #--------------------------------------------
        # Scene block
        _p_sdf, _p_h, _p_invs = sdf_decoder(tx_noc_p=tx_noc_p,
                                    win_a=self._deform_win_a, 
                                    t_code=t_code,
                                    amb_w=amb_w)

        #--------------------
        # in-bound check  
        _valid =(-1<=tx_noc_p)*(tx_noc_p<=1)
        _valid =reduce(_valid,'n c -> n', 'sum')
        va = _valid==3 

        #-----------------------------------------
        if self.skip_outter_samples:
            p_sdf = torch.ones_like(_p_sdf)
            p_sdf[va] = _p_sdf[va]
            
            p_h  = _p_h.clone().detach()
            p_h[va] = _p_h[va]

            p_invs = _p_invs.clone().detach()
            p_invs[va] = p_invs[va]
        else:
            p_sdf = _p_sdf
            p_h   = _p_h
            p_invs= _p_invs
        #-----------------------------------------
        if self.rgb_use_nv:
            assert tx_noc_p.requires_grad and _p_sdf.requires_grad
            _grad  = grad.gradient(_p_sdf, tx_noc_p)
            _p_nv  = torch.nn.functional.normalize(_grad,dim=-1)

            if self.skip_outter_samples:
                p_nv = _p_nv.clone().detach() 
                p_nv[va] = _p_nv[va]
            else:
                p_nv  = _p_nv
        else:
            p_nv = None
        

        #-----------------------------------------
        if sdf_only:
            out={ 
                'sdf':p_sdf,  
                'normal':p_nv,
                'p_invs':p_invs,
                'p_shift':p_shift,
                '_valid':va,
                }
            return out

        #-----------------------------------------
        # Color block (view-dependent)

        rgb_ft = p_h
        tk_ft  = p_h
        # rgb_ft = torch.cat((p_h,p_nv,tx_noc_p),dim=-1)

        if self.rgb_use_nv:
            rgb_ft = torch.cat((rgb_ft,p_nv),dim=-1)

        if self.use_bending:
            assert amb_w is not None
            rgb_ft = torch.cat((rgb_ft,amb_w),dim=-1)

        if self.rgb_use_viewdir:
            # 3*4*2=24
            # vdirs: f n 
            # 
            enc_vdirs = self.rgb_viewdir_encoder(vdirs, 
                                alpha=self._deform_win_a)

            if enc_vdirs.shape[0]!=rgb_ft.shape[0]:
                pdb.set_trace()
            rgb_ft = torch.cat((rgb_ft,enc_vdirs),dim=-1)

        #---------------
        _p_rgb = self.rgb_decoder(rgb_ft)  
        _p_rgb = torch.sigmoid(_p_rgb)

        if self.skip_outter_samples:
            p_rgb = torch.zeros_like(_p_rgb)
            p_rgb[va] = _p_rgb[va]
        else:
            p_rgb = _p_rgb

        #-----------------------------------------
        _p_tkft = self.tkft_decoder(tk_ft)   
        
        if self.skip_outter_samples:
            p_tkft = torch.zeros_like(_p_tkft)
            p_tkft[va] = _p_tkft[va]
        else:
            p_tkft = _p_tkft
            
        #-----------------------------------------
        out={ 
            'sdf':p_sdf, 
            'rgb':p_rgb,
            'tracker_ft':p_tkft,
            'normal':p_nv,
            'p_invs':p_invs,
            'p_shift':p_shift,
            '_valid':va,
            '_noc_p':noc_p,
            '_tx_noc_p':tx_noc_p,
        }

        return out



#-----------------------------------------------------------
class NeuS_MainMLP(nn.Module):

    def __init__(self,
                 rgb_in_ft_dim,
                 rgb_lnum,
                 rgb_hdim,
                 rgb_use_viewdir, 
                 rgb_use_nv,
                 rgb_use_wn,
                 skip_outter_samples,
                 append_t,
                 use_bending,
                 bend_wdim,
            ):

        super().__init__() 

        self.rgb_use_viewdir = rgb_use_viewdir
        self.rgb_use_nv = rgb_use_nv 

        self.append_t=append_t
        self.use_bending=use_bending

        self._deform_win_a=1.0
        self._deform_win_b=1.0
        
        self.skip_outter_samples = skip_outter_samples
        self.bend_wdim=bend_wdim # for rigid object awb=0

        #-------------------------------------------   
        # [ft,p,nv,view]
        # encodeing 
        #   ft: h
        #   p:  x
        #   nv: 3
        #   view: 3+2*4*3

        rgb_in_dim = rgb_in_ft_dim 

        if self.rgb_use_nv:
            rgb_in_dim +=3

        if self.use_bending:
            rgb_in_dim += bend_wdim

            _amb_w = torch.zeros((bend_wdim),dtype=torch.float32)
            nn.init.uniform_(_amb_w,0.0,1.0e-4)
            
            self.register_buffer('rigid_amb_w', _amb_w )

        if self.rgb_use_viewdir:

            # 3*4*2=24
            self.rgb_viewdir_encoder = WindowedPosEncoder(
                                            3,0,4, 
                                            use_identity=True)

            rgb_in_dim +=  self.rgb_viewdir_encoder.get_output_dim()

            
        self.rgb_decoder =make_mlp(
                                in_dim=rgb_in_dim, 
                                out_dim=3,
                                hidden_dim=rgb_hdim,
                                lnum=rgb_lnum,
                                act_type='relu',
                                act_params={},
                                last_linear=True,
                                use_wn=True ) 

        #---------------------------
        self.log_grad_module=[self.rgb_decoder]  


    def convert_coord(self, p, min_bound, grid_size):  
        assert p.ndim==2
        assert min_bound.ndim==2
        assert grid_size.ndim==2
        
        max_v = grid_size.max()
        # noc_p = ((p - min_bound)/grid_size)*2-1.0
        noc_p = ((p - min_bound)/max_v)*2-1.0
        return noc_p 

    def forward(self, p, vdirs, scene_data, normalize, motion_net, p_tidx, deform=False, sdf_only=False ):
        
        assert p.ndim==2 
        if vdirs is not None:
            assert p.shape==vdirs.shape, f'{p.shape} {vdirs.shape}'

        all_ft=[]
        all_valid=[]

        sdf_decoder = scene_data['model']
        grid_size = scene_data['grid_size']
        min_bound = scene_data['min_bound']
        
        # grid_size = sdf_decoder.grid_size
        # min_bound = sdf_decoder.min_bound

        #--------------------------------------------
        # normalize points
        if normalize:
            noc_p = self.convert_coord(p, min_bound, grid_size)
        else:
            noc_p = p 

        #--------------------------------------------
        # deformation block

        if deform:
            assert motion_net is not None
            assert p_tidx is not None
            assert p_tidx.shape[0] == noc_p.shape[0]

            if self.use_bending:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a,
                                    win_b=self._deform_win_b,)
                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = rt['w']

            else:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a )

                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = None

        else:
            p_shift = None 
            tx_noc_p= noc_p

            if self.use_bending:

                amb_w = repeat(self.rigid_amb_w, 
                              'c -> n c', n=noc_p.shape[0]).clone()

                #amb_w = torch.zeros(
                #            (noc_p.shape[0], self.bend_wdim),
                #            device=noc_p.device)
            else:
                amb_w = None

        #--------------------
        # n,c 
        if self.append_t :
            # n,t
            t_code = motion_net.get_time_code(p_tidx)
        else:
            t_code = None

        #--------------------------------------------        
        if tx_noc_p.requires_grad ==False:
            tx_noc_p.requires_grad=True
        
        #--------------------------------------------
        # Scene block
        _p_sdf, _p_h, _p_invs = sdf_decoder(tx_noc_p=tx_noc_p,
                                    win_a=self._deform_win_a, 
                                    t_code=t_code,
                                    amb_w=amb_w)

        #--------------------
        # in-bound check  
        _valid =(-1<=tx_noc_p)*(tx_noc_p<=1)
        _valid =reduce(_valid,'n c -> n', 'sum')
        va = _valid==3 

        #-----------------------------------------
        if self.skip_outter_samples:
            p_sdf = torch.ones_like(_p_sdf)
            p_sdf[va] = _p_sdf[va]
            
            p_h  = _p_h.clone().detach()
            p_h[va] = _p_h[va]

            p_invs = _p_invs.clone().detach()
            p_invs[va] = p_invs[va]
        else:
            p_sdf = _p_sdf
            p_h   = _p_h
            p_invs= _p_invs 

        #-----------------------------------------
        if sdf_only:
            out={ 
                'sdf':p_sdf,  
                #'normal':p_nv,
                'p_invs':p_invs,
                'p_shift':p_shift,
                '_valid':va,
                }
            return out
        
        #-----------------------------------------
        if self.rgb_use_nv:
            assert tx_noc_p.requires_grad and _p_sdf.requires_grad
            _grad  = grad.gradient(_p_sdf, tx_noc_p)
            _p_nv  = torch.nn.functional.normalize(_grad,dim=-1)

            if self.skip_outter_samples:
                p_nv = _p_nv.clone().detach() 
                p_nv[va] = _p_nv[va]
            else:
                p_nv  = _p_nv

            p_gard = _grad
        else:
            p_nv = None
            p_gard = None
        
        #-----------------------------------------
        # Color block (view-dependent)

        rgb_ft = p_h
        # rgb_ft = torch.cat((p_h,p_nv,tx_noc_p),dim=-1)

        if self.rgb_use_nv:
            rgb_ft = torch.cat((rgb_ft,p_nv),dim=-1)

        if self.use_bending:
            assert amb_w is not None
            rgb_ft = torch.cat((rgb_ft,amb_w),dim=-1)

        if self.rgb_use_viewdir:
            # 3*4*2=24
            # vdirs: f n 
            # 
            enc_vdirs = self.rgb_viewdir_encoder(vdirs, 
                                alpha=self._deform_win_a)

            if enc_vdirs.shape[0]!=rgb_ft.shape[0]:
                pdb.set_trace()
            rgb_ft = torch.cat((rgb_ft,enc_vdirs),dim=-1)

        #---------------
        _p_rgb = self.rgb_decoder(rgb_ft)  
        _p_rgb = torch.sigmoid(_p_rgb)

        if self.skip_outter_samples:
            p_rgb = torch.zeros_like(_p_rgb)
            p_rgb[va] = _p_rgb[va]
        else:
            p_rgb = _p_rgb
            
        #-----------------------------------------
        out={ 
            'sdf':p_sdf, 
            'rgb':p_rgb,
            'normal':p_nv,
            'p_gard':p_gard,
            'p_invs':p_invs,
            'p_shift':p_shift,
            '_valid':va,
            '_noc_p':noc_p,
            '_tx_noc_p':tx_noc_p,
        }
        return out

#-----------------------------------------------------------
class NeuS_MainMLP_v2(nn.Module):
    #
    # sdf_decoder return a output DICT
    # 
    def __init__(self,
                 rgb_in_ft_dim,
                 rgb_lnum,
                 rgb_hdim,
                 rgb_use_viewdir,  
                 rgb_use_wn,
                 sdf_predict_normal,
                 skip_outter_samples,
                 append_t,
                 use_bending,
                 bend_wdim,
            ):

        super().__init__() 

        self.rgb_use_viewdir = rgb_use_viewdir
        #self.rgb_use_nv = rgb_use_nv 

        self.append_t=append_t
        self.use_bending=use_bending

        self._deform_win_a=1.0
        self._deform_win_b=1.0
        
        self.skip_outter_samples = skip_outter_samples
        self.bend_wdim=bend_wdim # for rigid object awb=0

        self.sdf_predict_normal = sdf_predict_normal
        #-------------------------------------------   
        # [ft,p,nv,view]
        # encodeing 
        #   ft: h
        #   p:  x
        #   nv: 3
        #   view: 3+2*4*3

        rgb_in_dim = rgb_in_ft_dim 

        #if self.rgb_use_nv:
        #    rgb_in_dim +=3

        if self.use_bending:
            rgb_in_dim += bend_wdim

            _amb_w = torch.zeros((bend_wdim),dtype=torch.float32)
            nn.init.uniform_(_amb_w,0.0,1.0e-4)
            
            self.register_buffer('rigid_amb_w', _amb_w )

        if self.rgb_use_viewdir:

            # 3*4*2=24
            self.rgb_viewdir_encoder = WindowedPosEncoder(
                                            3,0,4, 
                                            use_identity=True)

            rgb_in_dim +=  self.rgb_viewdir_encoder.get_output_dim()

            
        self.rgb_decoder =make_mlp(
                                in_dim=rgb_in_dim, 
                                out_dim=3,
                                hidden_dim=rgb_hdim,
                                lnum=rgb_lnum,
                                act_type='relu',
                                act_params={},
                                last_linear=True,
                                use_wn=True ) 

        #---------------------------
        self.log_grad_module=[self.rgb_decoder]  


    def convert_coord(self, p, min_bound, grid_size):  
        assert p.ndim==2
        assert min_bound.ndim==2
        assert grid_size.ndim==2
        
        max_v = grid_size.max()
        # noc_p = ((p - min_bound)/grid_size)*2-1.0
        noc_p = ((p - min_bound)/max_v)*2-1.0
        return noc_p 
        
    def forward(self, p, vdirs, scene_data, normalize, motion_net, p_tidx, deform=False, sdf_only=False ):
        
        assert p.ndim==2 
        if vdirs is not None:
            assert p.shape==vdirs.shape, f'{p.shape} {vdirs.shape}'

        all_ft=[]
        all_valid=[]

        sdf_decoder = scene_data['model']
        grid_size = scene_data['grid_size']
        min_bound = scene_data['min_bound']
        
        # grid_size = sdf_decoder.grid_size
        # min_bound = sdf_decoder.min_bound

        #--------------------------------------------
        # normalize points
        if normalize:
            noc_p = self.convert_coord(p, min_bound, grid_size)
        else:
            noc_p = p 

        #--------------------------------------------
        # deformation block

        if deform:
            assert motion_net is not None
            assert p_tidx is not None
            assert p_tidx.shape[0] == noc_p.shape[0]

            if self.use_bending:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a,
                                    win_b=self._deform_win_b,)
                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = rt['w']

            else:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a )

                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = None

        else:
            p_shift = None 
            tx_noc_p= noc_p

            if self.use_bending:

                amb_w = repeat(self.rigid_amb_w, 
                              'c -> n c', n=noc_p.shape[0]).clone()

                #amb_w = torch.zeros(
                #            (noc_p.shape[0], self.bend_wdim),
                #            device=noc_p.device)
            else:
                amb_w = None

        #--------------------
        # n,c 
        if self.append_t :
            # n,t
            t_code = motion_net.get_time_code(p_tidx)
        else:
            t_code = None

        #--------------------------------------------        
        if tx_noc_p.requires_grad ==False:
            tx_noc_p.requires_grad=True
        
        #--------------------------------------------
        # Scene block
        sout  = sdf_decoder(tx_noc_p=tx_noc_p,
                            win_a=self._deform_win_a, 
                            t_code=t_code,
                            amb_w=amb_w)

        _p_sdf = sout['sdf']
        _p_h   = sout['ft']
        _p_invs= sout['invs']

        if self.sdf_predict_normal:
            _p_nv  = sout['nv']

        #--------------------
        # in-bound check  
        _valid =(-1<=tx_noc_p)*(tx_noc_p<=1)
        _valid =reduce(_valid,'n c -> n', 'sum')
        va = _valid==3 

        #-----------------------------------------
        if self.skip_outter_samples:
            p_sdf = torch.ones_like(_p_sdf)
            p_sdf[va] = _p_sdf[va]
            
            p_h  = _p_h.clone().detach()
            p_h[va] = _p_h[va]

            p_invs = _p_invs.clone().detach()
            p_invs[va] = p_invs[va]

            if self.sdf_predict_normal:
                p_nv = _p_nv.clone().detach()
                p_nv[va] = _p_nv[va]
        else:
            p_sdf = _p_sdf
            p_h   = _p_h
            p_invs= _p_invs 

            if self.sdf_predict_normal:
                p_nv  = _p_nv


        if not self.sdf_predict_normal:
            p_nv = None

        #-----------------------------------------
        if sdf_only:
            out={ 
                'sdf':p_sdf,  
                'normal':p_nv,
                'p_invs':p_invs,
                'p_shift':p_shift,
                '_valid':va,
                'p_h':p_h,
                }
            return out
        
        #-----------------------------------------
        # Color block (view-dependent)

        rgb_ft = p_h

        #if self.rgb_use_nv:
        #    rgb_ft = torch.cat((rgb_ft,p_nv),dim=-1)

        if self.use_bending:
            assert amb_w is not None
            rgb_ft = torch.cat((rgb_ft,amb_w),dim=-1)

        if self.rgb_use_viewdir:
            # 3*4*2=24
            # vdirs: f n 
            # 
            enc_vdirs = self.rgb_viewdir_encoder(vdirs, 
                                alpha=self._deform_win_a)

            if enc_vdirs.shape[0]!=rgb_ft.shape[0]:
                pdb.set_trace()
            rgb_ft = torch.cat((rgb_ft,enc_vdirs),dim=-1)

        #---------------
        _p_rgb = self.rgb_decoder(rgb_ft)  
        _p_rgb = torch.sigmoid(_p_rgb)

        if self.skip_outter_samples:
            p_rgb = torch.zeros_like(_p_rgb)
            p_rgb[va] = _p_rgb[va]
        else:
            p_rgb = _p_rgb
            
        #-----------------------------------------
        out={ 
            'sdf':p_sdf, 
            'rgb':p_rgb,
            'normal':p_nv,
            'p_invs':p_invs,
            'p_shift':p_shift,
            '_valid':va,
            '_noc_p':noc_p,
            '_tx_noc_p':tx_noc_p,
        }
        return out

