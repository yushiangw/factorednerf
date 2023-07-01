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

from .blocks import make_mlp, SkipCMLP

from .time import TimeCodes
from .rigid import SE3_OptBlock 

from .deform import WarpField, BendMLP, BijMap

from ..rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
#------------------------------

class RigidPoses(nn.Module):

    def __init__(self, frame_num, optimize_pose, use_cano_frame, cano_idx): 

        super().__init__()

        self.poses = nn.ModuleList()
        self.frame_num = frame_num

        # for k in range(frame_num): 
            # se = SE3_OptBlock(use_parameter=optimize_pose)
            # self.poses.append(se) 

        for k in range(frame_num): 
            if optimize_pose and use_cano_frame and k==cano_idx:
                opt = False 
            else:
                opt = optimize_pose

            se = SE3_OptBlock(use_parameter=opt)

            self.poses.append(se) 

    def get_params_by_flag(self, rigid=False ):

        params =[]
        if rigid:
            params += self.poses.parameters()
        return params
        
    def get_frame_num(self):
        return self.frame_num

    def ini_rigid_pose(self, t_idx, mat44):
        assert mat44.ndim==2
        quat = matrix_to_quaternion(mat44[:3,:3])
        trsl = mat44[:3,3].squeeze()
        self.poses[t_idx].reset(quat,trsl)

        re_mat44=self.poses[t_idx]()['RT'].to(mat44.device)

        err = (re_mat44-mat44).abs().max()
        assert err <  1.0e-3, err

    def is_rigid(self):
        return True  

    def apply_rigid(self, t_idx, p, apply_trsl):  
        assert p.ndim==2 
        pose = self.poses[t_idx]        
        q = pose.apply(p, apply_trsl)   
        return q

    def apply_inv(self, t_idx, p, apply_trsl):
        assert p.ndim==2  
        pose = self.poses[t_idx] 
        q = pose.apply_inv(p, apply_trsl) 
        return q

    def get_rigid_pose(self, t_idx): 
        pose_mat = self.poses[t_idx]()['RT'][0]
        return pose_mat

    def get_all_rigid_poses(self):
        rot_list = []
        for k in range(len(self.poses)):
            pose_mat = self.poses[k]()['RT'][0]
            rot_list.append(pose_mat)
        RTs = torch.stack(rot_list)
        return RTs

    def forward(self):
        raise Exception('unimplement') 

#------------------------------
class RigidPoses_wTCodes(nn.Module):

    def __init__(self, frame_num, t_dim, optimize_pose, use_cano_frame, cano_idx): 

        super().__init__()

        self.poses = nn.ModuleList()

        self.frame_num = frame_num
        self.cano_idx = cano_idx

        for k in range(frame_num): 
            if optimize_pose and use_cano_frame and k==cano_idx:
                opt = False 
            else:
                opt = optimize_pose

            se = SE3_OptBlock(use_parameter=opt)

            self.poses.append(se) 

        self.tblock = TimeCodes(frame_num, t_dim) 

    def get_params_by_flag(self, tcode=False, rigid=False ):

        params =[]
        if tcode:
            params += self.tblock.parameters()
        if rigid:
            params += self.poses.parameters()

        return params

    def get_frame_num(self):
        return self.frame_num

    def ini_rigid_pose(self, t_idx, mat44):
        assert mat44.ndim==2

        quat = matrix_to_quaternion(mat44[:3,:3]) 
        trsl = mat44[:3,3].squeeze()
        
        self.poses[t_idx].reset(quat,trsl)

        re_mat44=self.poses[t_idx]()['RT'].to(mat44.device)

        err = (re_mat44-mat44).abs().max()
           
        assert err < 1.0e-3, err 

    def is_rigid(self):
        return True  

    def get_time_code(self, t_idx):
        return self.tblock(t_idx) 

    def apply_rigid(self, t_idx, p, apply_trsl):  
        assert p.ndim==2 
        pose = self.poses[t_idx]        
        q = pose.apply(p, apply_trsl)   
        return q

    def apply_inv(self, t_idx, p, apply_trsl):
        assert p.ndim==2  
        pose = self.poses[t_idx] 
        q = pose.apply_inv(p, apply_trsl) 
        return q

    def get_rigid_pose(self, t_idx): 
        pose_mat = self.poses[t_idx]()['RT'][0]
        return pose_mat

    def get_all_rigid_poses(self):

        rot_list = []
        for k in range(len(self.poses)):
            pose_mat = self.poses[k]()['RT'][0]
            rot_list.append(pose_mat)

        RTs = torch.stack(rot_list)

        return RTs

    def forward(self):
        raise Exception('unimplement') 


#------------------------------
class RigidPoses_WarpField_wTCodes(nn.Module):

    def __init__(self, frame_num, optimize_rigid_pose, 
            t_dim, warp_use_encoding, 
            warp_min_deg, warp_max_deg,
            use_cano_frame, cano_idx ):

        super().__init__()

        self.use_cano_frame = use_cano_frame
        self.cano_idx = cano_idx

        if self.use_cano_frame:
            assert self.cano_idx!=-1 and self.cano_idx>=0 
            assert self.cano_idx is not None 

        self.frame_num = frame_num

        self.tblock = TimeCodes(frame_num, t_dim) 

        
        self.rigid_poses = RigidPoses(frame_num, optimize_rigid_pose,                    use_cano_frame=use_cano_frame,
                                cano_idx=cano_idx)

        self.dmlp = WarpField( in_dim=3, 
                        use_pos_encoding=warp_use_encoding, 
                        min_deg=warp_min_deg, 
                        max_deg=warp_max_deg,  z_dim=t_dim )

        self.log_grad_module=[self.dmlp]
    
    def get_params_by_flag(self, tcode=False, rigid=False, deform=False):

        params =[]

        if tcode:
            params += self.tblock.parameters()

        if rigid:
            params += self.rigid_poses.parameters()

        if deform:
            params += self.dmlp.parameters()
    
        return params

    def get_params(self, only_deform_net):

        params =[]
        params += self.tblock.parameters()
        params += self.dmlp.parameters()

        if not only_deform_net: 
            params += self.rigid_poses.parameters()

        return params
        
    def get_frame_num(self):
        return self.frame_num

    def is_rigid(self):
        return False

    def ini_rigid_pose(self, t_idx, mat44):

        self.rigid_poses.ini_rigid_pose(t_idx, mat44)

    def get_rigid_pose(self, t_idx):  
        return self.rigid_poses.get_rigid_pose(t_idx)

    def get_all_rigid_poses(self):

        return self.rigid_poses.get_all_rigid_poses()

    def get_time_code(self, t_idx):

        return self.tblock(t_idx) 

    def apply_rigid(self, t_idx, p, apply_trsl): 
        assert p.ndim==2
        p2 = self.rigid_poses.apply_rigid(t_idx, p, apply_trsl) 
        
        return p2  

    def apply_deform_batch(self, t_idx, p, win_a):

        assert t_idx.shape[0] == p.shape[0] and t_idx.ndim==1
        assert p.ndim ==2  

        # n c
        tz = self.get_time_code(t_idx) 
        
        q = self.dmlp(p=p, z=tz, win_a=win_a)  

        if self.use_cano_frame:
            cano_mask = (t_idx == self.cano_idx) 

            q2 = torch.zeros_like(p)
            q2[cano_mask] =p[cano_mask] # no deformation
            q2[~cano_mask]=q[~cano_mask]
            q=q2

        delta = q-p 

        rt ={
            'q':q,
            'delta':delta,
        }
        return rt 


    def apply_deform(self, t_idx, p, win_a,):

        assert p.ndim==2
        assert torch.is_tensor(t_idx) and t_idx.ndim<2

        t_idx2 = repeat(t_idx, ' -> n', n=p.shape[0])

        return self.apply_deform_batch(
                t_idx=t_idx2, p=p,  win_a=win_a)


    def apply_inv(self, t_idx, p, apply_trsl): 

        raise Exception('unimplement') 

    def forward(self):

        raise Exception('unimplement')

        return 

#------------------------------
class RigidPoses_WarpFieldBend_wTCodes(nn.Module):

    def __init__(self, frame_num, optimize_rigid_pose, 
            t_dim, warp_use_encoding, 
            warp_min_deg, warp_max_deg,
            bend_amb_dim, bend_min_deg, bend_max_deg,
            use_cano_frame, cano_idx ):

        super().__init__()


        self.use_cano_frame = use_cano_frame
        self.cano_idx = cano_idx

        if self.use_cano_frame:
            assert self.cano_idx!=-1 and self.cano_idx>=0 
            assert self.cano_idx is not None 

        self.frame_num = frame_num

        self.tblock = TimeCodes(frame_num, t_dim) 

        
        self.rigid_poses = RigidPoses(frame_num, optimize_rigid_pose,                    use_cano_frame=use_cano_frame,
                                cano_idx=cano_idx)


        self.dmlp = WarpField( in_dim=3, 
                        use_pos_encoding=warp_use_encoding, 
                        min_deg=warp_min_deg, 
                        max_deg=warp_max_deg,  z_dim=t_dim )


        self.bend_mlp = BendMLP( in_dim=3, 
                                 use_pos_encoding=True, 
                                 min_deg=bend_min_deg, 
                                 max_deg=bend_max_deg,
                                 z_dim=t_dim, 
                                 w_dim=bend_amb_dim)

        self.log_grad_module=[self.dmlp, self.bend_mlp]
    
    def get_params_by_flag(self, tcode=False, rigid=False, deform=False):

        params =[]

        if tcode:
            params += self.tblock.parameters()

        if rigid:
            params += self.rigid_poses.parameters()

        if deform:
            params += self.dmlp.parameters()
    
        return params

    def get_params(self, only_deform_net):

        params =[]
        params += self.tblock.parameters()
        params += self.dmlp.parameters()

        if not only_deform_net: 
            params += self.rigid_poses.parameters()

        return params
        
    def get_frame_num(self):
        return self.frame_num

    def is_rigid(self):
        return False

    def ini_rigid_pose(self, t_idx, mat44):

        self.rigid_poses.ini_rigid_pose(t_idx, mat44)

    def get_rigid_pose(self, t_idx):  
        return self.rigid_poses.get_rigid_pose(t_idx)

    def get_all_rigid_poses(self):

        return self.rigid_poses.get_all_rigid_poses()

    def get_time_code(self, t_idx):

        return self.tblock(t_idx) 

    def apply_rigid(self, t_idx, p, apply_trsl): 
        assert p.ndim==2
        p2 = self.rigid_poses.apply_rigid(t_idx, p, apply_trsl) 
        
        return p2  

    def apply_deform_batch(self, t_idx, p, win_a, win_b):

        assert t_idx.shape[0] == p.shape[0] and t_idx.ndim==1
        assert p.ndim ==2  

        # n c
        tz = self.get_time_code(t_idx) 
        
        q = self.dmlp(p=p, z=tz, win_a=win_a)  

        w = self.bend_mlp(p=p, z=tz, win_a=win_b)  

        if self.use_cano_frame:
            cano_mask = (t_idx == self.cano_idx)

            w2 = torch.zeros_like(w)
            w2[~cano_mask] = w[~cano_mask]

            q2 = torch.zeros_like(p)
            q2[cano_mask] =p[cano_mask] # no deformation
            q2[~cano_mask]=q[~cano_mask]
            
            q=q2
            w=w2

        delta = q-p 


        rt ={
            'q':q,
            'delta':delta,
            'w':w,
        }
        return rt 


    def apply_deform(self, t_idx, p, win_a,):

        assert p.ndim==2
        assert torch.is_tensor(t_idx) and t_idx.ndim<2

        t_idx2 = repeat(t_idx, ' -> n', n=p.shape[0])

        return self.apply_deform_batch(
                t_idx=t_idx2, p=p,  win_a=win_a)


    def apply_inv(self, t_idx, p, apply_trsl): 

        raise Exception('unimplement') 

    def forward(self):

        raise Exception('unimplement')

        return 


#------------------------------
class RigidPoses_BijMap_wTCodes(nn.Module):

    def __init__(self, frame_num, optimize_rigid_pose, bij_use_wn, t_dim, uv_min_deg, uv_max_deg, w_min_deg, w_max_deg,
            use_cano_frame, cano_idx ):

        super().__init__()

        self.use_cano_frame = use_cano_frame
        self.cano_idx = cano_idx

        if self.use_cano_frame:
            assert self.cano_idx!=-1 and self.cano_idx>=0 
            assert self.cano_idx is not None 

        self.frame_num = frame_num

        self.tblock = TimeCodes(frame_num, t_dim) 

        
        self.rigid_poses = RigidPoses(frame_num, optimize_rigid_pose,                          use_cano_frame=use_cano_frame,
                                      cano_idx=cano_idx)


        #=================================================== 

        self.dmlp = BijMap( use_wn=bij_use_wn,
                            z_dim=t_dim, 
                            uv_min_deg=uv_min_deg, 
                            uv_max_deg=uv_max_deg,  
                            w_min_deg=w_min_deg, 
                            w_max_deg=w_max_deg)

        #===================================================
        self.log_grad_module=[self.dmlp]
    
    def get_params_by_flag(self, tcode=False, rigid=False, deform=False):

        params =[]

        if tcode:
            params += self.tblock.parameters()

        if rigid:
            params += self.rigid_poses.parameters()

        if deform:
            params += self.dmlp.parameters()
    
        return params

    def get_params(self, only_deform_net):

        params =[]
        params += self.tblock.parameters()
        params += self.dmlp.parameters()

        if not only_deform_net: 
            params += self.rigid_poses.parameters()

        return params
        
    def get_frame_num(self):
        return self.frame_num

    def is_rigid(self):
        return False

    def ini_rigid_pose(self, t_idx, mat44):

        self.rigid_poses.ini_rigid_pose(t_idx, mat44)

    def get_rigid_pose(self, t_idx):  
        return self.rigid_poses.get_rigid_pose(t_idx)

    def get_all_rigid_poses(self):

        return self.rigid_poses.get_all_rigid_poses()

    def get_time_code(self, t_idx):

        return self.tblock(t_idx) 

    def apply_rigid(self, t_idx, p, apply_trsl): 
        assert p.ndim==2
        p2 = self.rigid_poses.apply_rigid(t_idx, p, apply_trsl) 
        
        return p2  

    def apply_deform_batch(self, t_idx, p, win_a ):

        assert t_idx.shape[0] == p.shape[0] and t_idx.ndim==1
        assert p.ndim ==2  

        # n c
        tz = self.get_time_code(t_idx) 
        
        q = self.dmlp(p=p, z=tz, win_a=win_a)  

        if self.use_cano_frame:
            cano_mask = (t_idx == self.cano_idx)

            #w2 = torch.zeros_like(w)
            #w2[~cano_mask] = w[~cano_mask] # attach deformation 

            q2 = torch.zeros_like(p)
            q2[cano_mask] =p[cano_mask] # no deformation
            q2[~cano_mask]=q[~cano_mask]# attach deformation 
            
            q=q2
            #w=w2
            
        delta = q-p 
        
        rt ={
            'q':q,
            'delta':delta,
        }
        return rt 


    def apply_deform(self, t_idx, p, win_a, ): 
        assert p.ndim==2
        assert torch.is_tensor(t_idx) and t_idx.ndim<2

        t_idx2 = repeat(t_idx, ' -> n', n=p.shape[0])

        return self.apply_deform_batch(
                t_idx=t_idx2, p=p,  win_a=win_a)


    def apply_inv(self, t_idx, p, apply_trsl): 

        raise Exception('unimplement') 

    def forward(self):

        raise Exception('unimplement')

        return 


class RigidPoses_BijMapBend_wTCodes(nn.Module):

    def __init__(self, frame_num, optimize_rigid_pose, bij_use_wn,
            t_dim, uv_min_deg, uv_max_deg, w_min_deg, w_max_deg, 
            bend_amb_dim, bend_min_deg, bend_max_deg,
            use_cano_frame, cano_idx):

        super().__init__()

        self.use_cano_frame = use_cano_frame
        self.cano_idx = cano_idx

        if self.use_cano_frame:
            assert self.cano_idx!=-1 and self.cano_idx>=0 
            assert isinstance(self.cano_idx,int)

        self.frame_num = frame_num

        self.tblock = TimeCodes(frame_num, t_dim) 

        
        self.rigid_poses = RigidPoses(frame_num, optimize_rigid_pose, 
                                        use_cano_frame=use_cano_frame,
                                        cano_idx=cano_idx)


        #=================================================== 

        self.dmlp = BijMap( use_wn=bij_use_wn,
                            z_dim=t_dim, 
                            uv_min_deg=uv_min_deg, 
                            uv_max_deg=uv_max_deg,  
                            w_min_deg=w_min_deg, 
                            w_max_deg=w_max_deg )

        self.bend_mlp = BendMLP( in_dim=3, 
                                 use_pos_encoding=True, 
                                 min_deg=bend_min_deg, 
                                 max_deg=bend_max_deg,
                                 z_dim=t_dim, 
                                 w_dim=bend_amb_dim)

        #===================================================
        self.log_grad_module=[self.dmlp, self.bend_mlp]
    
    def get_params_by_flag(self, tcode=False, rigid=False, deform=False):

        params =[]

        if tcode:
            params += self.tblock.parameters()

        if rigid:
            params += self.rigid_poses.parameters()

        if deform:
            params += self.dmlp.parameters()
    
        return params

    def get_params(self, only_deform_net):

        params =[]
        params += self.tblock.parameters()
        params += self.dmlp.parameters()

        if not only_deform_net: 
            params += self.rigid_poses.parameters()

        return params
        
    def get_frame_num(self):
        return self.frame_num

    def is_rigid(self):
        return False

    def ini_rigid_pose(self, t_idx, mat44):

        self.rigid_poses.ini_rigid_pose(t_idx, mat44)

    def get_rigid_pose(self, t_idx):  
        return self.rigid_poses.get_rigid_pose(t_idx)
        
    def get_all_rigid_poses(self):

        return self.rigid_poses.get_all_rigid_poses()

    def get_time_code(self, t_idx):

        return self.tblock(t_idx) 

    def apply_rigid(self, t_idx, p, apply_trsl): 
        assert p.ndim==2
        p2 = self.rigid_poses.apply_rigid(t_idx, p, apply_trsl) 
        
        return p2  


    def apply_deform_batch(self, t_idx, p, win_a, win_b):

        assert t_idx.shape[0] == p.shape[0] and t_idx.ndim==1
        assert p.ndim ==2  

        # n c
        tz = self.get_time_code(t_idx) 

        q = self.dmlp(p=p, z=tz, win_a=win_a)  

        w = self.bend_mlp(p=p, z=tz, win_a=win_b)  

        if self.use_cano_frame:
            cano_mask = (t_idx == self.cano_idx)

            w2 = torch.zeros_like(w)
            w2[~cano_mask] = w[~cano_mask]

            q2 = torch.zeros_like(p)
            q2[cano_mask] =p[cano_mask]
            q2[~cano_mask]=q[~cano_mask]
            
            q=q2
            w=w2

        delta = q-p


        rt ={
            'q':q,
            'delta':delta,
            'w':w 
        }
        return rt 


    def apply_deform(self, t_idx, p, win_a, win_b ): 
        assert p.ndim==2
        assert torch.is_tensor(t_idx) and t_idx.ndim<2

        t_idx2 = repeat(t_idx, ' -> n', n=p.shape[0])

        return self.apply_deform_batch(
                t_idx=t_idx2, p=p,  win_a=win_a, win_b=win_b)


    def apply_inv(self, t_idx, p, apply_trsl): 

        raise Exception('unimplement') 

    def forward(self):

        raise Exception('unimplement')

        return 

