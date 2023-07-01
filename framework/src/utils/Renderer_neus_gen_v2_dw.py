import torch
import pdb 
from src.common import get_rays
from tqdm import tqdm
from einops import rearrange, reduce, repeat
import pdb 

class Renderer(object):
    def __init__(self, H, W, fx, fy, cx, cy,
        sample_module, dw_step=1,
        points_batch_size=128*4, ray_batch_size=1000):

        self.ray_batch_size    = ray_batch_size
        self.points_batch_size = points_batch_size

        #self.bound = bound 
        self.H, self.W, self.fx, self.fy, self.cx, self.cy =(H,W,fx,fy,cx,cy)

        self.sample_module= sample_module

        self.dw_step=dw_step
        assert dw_step>=1
        

    def render_img(self, obj_sce_data_list, obj_bounds_world, obj_idx2tidx,  f_idx, decoder, obj_kf_poses,  device, near_z, far_z,  render_sample_mode, sample_num, gt_depth,  render_shading=False, render_obj_vec=False, new_dw_step=None, show_prog=False):

        if new_dw_step is not None and new_dw_step>=1:
            self.dw_step=new_dw_step

        H = self.H
        W = self.W

        c2w = torch.eye(4,device=device)

        rays_o, rays_d = get_rays(H, W, self.fx, self.fy, 
                            self.cx, self.cy,  c2w, device) 

        if self.dw_step>1: 
            H = H//self.dw_step
            W = W//self.dw_step

            rays_o=rays_o[::self.dw_step,::self.dw_step]
            rays_d=rays_d[::self.dw_step,::self.dw_step]

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # rays_o = pose_network.apply_rigid(t_idx=t_idx, p=rays_o, apply_trsl=True)
        # rays_d = pose_network.apply_rigid(t_idx=t_idx, p=rays_d, apply_trsl=False)

        depth_list = []
        uncertainty_list = []
        color_list = []
        shading_list = []
        obj_alpha_list   = []
        objvec_list = []
        dv_list=[]

        ray_batch_size = self.ray_batch_size

        pbar = range(0, rays_d.shape[0], ray_batch_size)

        if show_prog:
            pbar = tqdm(pbar)

        for i in pbar:

            b_rays_o = rays_o[i:i+ray_batch_size]
            b_rays_d = rays_d[i:i+ray_batch_size]

            # F,N,C
            b_rays_o = b_rays_o.unsqueeze(0)
            b_rays_d = b_rays_d.unsqueeze(0)

            # def render_rays_w_objs( 
            #     cam_rays_o, cam_rays_d, f_indices, obj_idx2tidx,
            #     obj_kf_poses, obj_bounds_world, obj_sce_data_list, 
            #     decoder, render_mode, render_sample_mode, render_sample_num, 
            #     near_z, far_z, device, verbose=False):
            # 
            # render_mode=render_mode,
            # 
            
            rt = self.sample_module.render_rays_w_objs(
                            cam_rays_o=b_rays_o, 
                            cam_rays_d=b_rays_d, 
                            f_indices=[f_idx], 
                            obj_idx2tidx=obj_idx2tidx,
                            obj_kf_poses=obj_kf_poses, 
                            obj_bounds_world=obj_bounds_world,
                            obj_sce_data_list=obj_sce_data_list, 
                            decoder=decoder, 
                            render_sample_mode=render_sample_mode,
                            render_sample_num=sample_num,
                            near_z=near_z,
                            far_z=far_z,
                            esti_normal=render_shading,
                            render_obj_vec=render_obj_vec,
                            device=device) 

            depth = rt['depth'].squeeze(0)
            color = rt['color'].squeeze(0)
            uncertainty = rt['uncertainty'].squeeze(0)
            
            debug_valid = rt['debug_valid'].sum(dim=-1).squeeze(0)

            # k f n m
            obj_weights = rt['obj_weights'] 
            # k f n
            obj_alpha = obj_weights.sum(dim=-1)
            obj_alpha = rearrange(obj_alpha, ' k f n -> f n k')


            depth = depth.detach().cpu()
            uncertainty = uncertainty.detach().cpu()
            color = color.detach().cpu()
            obj_alpha = obj_alpha.detach().cpu()

            dv = debug_valid.detach().cpu()

            depth_list.append(depth)
            uncertainty_list.append(uncertainty)
            color_list.append(color) 
            obj_alpha_list.append(obj_alpha)
            dv_list.append(dv)

            if render_shading:
                # n k c
                sh = rt['shading'].squeeze(0).detach().cpu()
                shading_list.append(sh)  

            if render_obj_vec:
                # f n k
                ovec = rt['objvec'].detach().cpu()
                objvec_list.append(ovec)
                #pred_objvec = rearrange(pred_objvec,'f n k -> k f n') 

        depth = torch.cat(depth_list, dim=0)#.double()
        uncertainty = torch.cat(uncertainty_list, dim=0)#.double()
        color = torch.cat(color_list, dim=0)
        obj_alpha = torch.cat(obj_alpha_list, dim=0)

        depth = depth.reshape(H, W)
        uncertainty = uncertainty.reshape(H, W)
        color = color.reshape(H, W, 3)
        obj_alpha = obj_alpha.reshape(H, W, -1)
        
        debug_valid_img = torch.cat(dv_list, dim=0)#.double()
        debug_valid_img = debug_valid_img.reshape(H, W, -1) 
        
        if render_obj_vec:
            obj_vec = torch.cat(objvec_list,dim=0)
            obj_vec = obj_vec.reshape(H, W, -1)
        else:
            obj_vec = None

        if render_shading:
            sh = torch.cat(shading_list, dim=0)
            sh = sh.reshape(H,W,3)
        else:
            sh = None

        rt ={
            'depth':depth,
            'color':color,
            'shading':sh,
            'obj_alpha':obj_alpha,
            'obj_vec':obj_vec,
        }
        return rt 

