import os
import time
import pdb 
import argparse 
import copy
import logging 
import gc 
from collections import defaultdict, OrderedDict
from types import SimpleNamespace
from functools import partial
import pickle
#=========================================================+

import torch
import torch.multiprocessing
import torch.multiprocessing as mp
import torch.nn.functional 
import torch.nn as nn

import numpy as np
from scipy.spatial.transform import Rotation as R
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm
import cv2

import matplotlib
import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg') 

from types import SimpleNamespace
import imageio
import skimage.io
from skimage.io import imsave

#=========================================================+
import src.metrics 

from src import config 
from src.coord_tools import valid_backproj, world2image_np, camera2image

from src.common import torch_load, random_nd_index, plot_loss, plot_loss_v2, save_ray_log, count_parameters

from src.common import  get_samples_v3_my, get_samples_v4_my,  get_surface_sample_v2
 
from src.my_model import grad  
from src.timer import MyTimer

from src.utils import Renderer_neus_gen_v1, Renderer_neus_gen_v2_dw

from src import writer 
from src import occ_tools_v2, nerf_tools_v3, neus_tools_v8

#=========================================================

from src.optvis_tools  import (vis_geo_and_save_render_mask,
    vis_and_save_alpha, make_video, make_gif,
    save_depth_vis, plot_grad_log )

from src.filter import depth_filter,dilate
from src.dyn_sampler import ray_box_intersection

#=========================================================

torch.multiprocessing.set_sharing_strategy('file_system')

clog = logging.getLogger('')
clog.setLevel(logging.DEBUG)

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


script_name = os.path.basename(__file__).split('.')[0]


class Helper():
    def __init__(self, cfg, args, cfg_name, no_dir=False):

        self.cfg  = cfg
        self.args = args
        self.device = torch.device('cuda:0')
        
        assert script_name==cfg['script_name']
        
        self.dataset_name = cfg['dataset']
        self.verbose      = cfg['verbose']

        self.output = os.path.join(cfg['data']['output_root'], cfg_name)

        self.update_cam()

        # ======================================================
        self.sce_type = cfg['sce_type']

        if self.sce_type == 'neus_v8':
            #self.render_moudle = Renderer_neus_gen_v1
            self.render_moudle = Renderer_neus_gen_v2_dw

            self.sample_moudle = neus_tools_v8
            self.render_shading = True
        
        elif self.sce_type == 'occ_v2':
            #self.render_moudle = Renderer_neus_gen_v1
            self.render_moudle = Renderer_neus_gen_v2_dw
            self.sample_moudle = occ_tools_v2
            self.render_shading = False

        elif self.sce_type == 'nerf_v3':
            #self.render_moudle  = Renderer_neus_gen_v1
            self.render_moudle = Renderer_neus_gen_v2_dw
            
            self.sample_moudle  = nerf_tools_v3
            self.render_shading = False
        else:
            raise Exception('unknown scene type:'+self.sce_type)
        

        #===========================
        db_mod   = __import__(cfg['dataset']['module'], 
                              fromlist=[cfg['dataset']['class']])
        db_class  = getattr(db_mod, cfg['dataset']['class'])

        # cam_cfg, input_folder, device
        self.frame_reader_train = db_class(cfg['dataset'], cfg['cam'], cfg['dataset']['train'] , self.device) 
        
        self.frame_reader_valid = db_class(cfg['dataset'], cfg['cam'], cfg['dataset']['valid'] , self.device) 

        # tkft_train_fp=os.path.join(cfg['dataset']['train'] ,'track_ft.pkl')
        # tkft_val_fp  =os.path.join(cfg['dataset']['valid'] ,'track_ft.pkl')

        # with open(tkft_train_fp,'rb') as fin:
        #     x=pickle.load(fin)['siamfc_ft']
        #     # b c h w
        #     self.tracker_ft_train = x

        # with open(tkft_val_fp,'rb') as fin:
        #     x=pickle.load(fin)['siamfc_ft']
        #     # b c h w
        #     self.tracker_ft_val = x

        #===========================
        self.shared_cfg=SimpleNamespace(**cfg['shared'])

        #===========================
        if  self.shared_cfg.opt_frames_num==-1:
            fnum = len(self.frame_reader_train)
        else:
            fnum = self.shared_cfg.opt_frames_num
        
        # scfg.obj_frames_indices
        self.opt_frames_indices = [i for i in range(fnum)]

        self.opt_eval_fidx_step = self.shared_cfg.opt_eval_fidx_step

        #===========================
        self.opt_mode  = cfg['opt_2']['opt_mode'] 

        self.opt_cfgs_dict={}

        for mm in self.opt_mode:
           self.opt_cfgs_dict[mm] = SimpleNamespace(**cfg['opt_2'][mm])  

        #======================================================
        # obj_bd_world=np.asarray(self.shared_cfg.obj_bounds_world)
        # obj_bd_world=torch.from_numpy(np.array(obj_bd_world)).to(torch.float32)

        # # K,2,3 
        # obj_bounds_world = obj_bd_world

        # obj_bd_cam=np.asarray(self.shared_cfg.obj_bounds_camera)
        # obj_bd_cam=torch.from_numpy(np.array(obj_bd_cam)).to(torch.float32)

        # # K,2,3 
        # obj_bounds_camera = obj_bd_cam

        #===========================================================
        scfg = self.shared_cfg

        scfg.g_obj_num = len(scfg.g_obj_uids)
        
        pinit_m = self.shared_cfg.pose_init_mode

        if pinit_m in [ 'noise_GT_t2', '1stframe' ]: 
            raise Exception('unsupported')

        elif pinit_m in ['GT', 'eye','est_pose_v1']: 

            if scfg.bbox_mode=='siammask_icp':
                obj_bounds = self.frame_reader_train.load_est_bbox()
            elif scfg.bbox_mode=='anno':
                obj_bounds = self.frame_reader_train.load_gt_bbox()
            else:
                raise Exception('unsupported bbox mode:'+scfg.bbox_mode)
        else:
            raise Exception('unsupported')

        scfg.g_obj_bounds = []

        for u in scfg.g_obj_uids:
            scfg.g_obj_bounds.append(obj_bounds[u])

        scfg.g_obj_bounds = torch.stack(scfg.g_obj_bounds,dim=0)
        scfg.g_obj_bounds = scfg.g_obj_bounds.to(self.device)

        #===========================

        obj_idx2t=[]
        for k in range(scfg.g_obj_num):
            k_idx2t={}

            for t,idx in enumerate(self.opt_frames_indices):
                k_idx2t[idx]=t
            obj_idx2t.append(k_idx2t)

        scfg.g_obj_idx2t = obj_idx2t

        #====================================================== 

        self.shared_cfg.ray_sample_mode = self.sample_moudle.get_sample_mode(
                                              self.shared_cfg.ray_sample_mode)


        #======================================================
        self.rigid_posenet_spec    = cfg['rigid_posenet'] 
        self.nonrigid_posenet_spec = cfg['nonrigid_posenet'] 


        self.ignore_edge_W = 0 #cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = 0 #cfg['tracking']['ignore_edge_H']

        #======================================================
        # build model and decoder 
        def sce_factory_func(min_bound, max_bound, ray_step_size, overwrite_args, spec ):
            
            sp2 = copy.deepcopy(spec) 

            if overwrite_args is not None:
                for k,v in overwrite_args.items():
                    sp2['args'][k]=v
                    print(f'[overwrite_args][{k}]={v}')

            cm = config.get_model(sp2)
            # cm.initial( min_bound=min_bound, max_bound=max_bound ) 

            grid_size = max_bound-min_bound
            rt={
                'model':cm, 
                'min_bound':min_bound,
                'ray_step_size':ray_step_size,
                'grid_size':grid_size,
            }

            return rt

        self.sce_factory = partial(sce_factory_func, 
                                   spec=cfg['sce_model'] )

        self.network = config.get_model(cfg['network']).to(self.device) 

          
        dw_step=cfg['render_dw_step'] 
        self.renderer = self.render_moudle.Renderer(
                         H=self.H, 
                         W=self.W, 
                         fx=self.fx,
                         fy=self.fy, 
                         cx=self.cx,
                         cy=self.cy,
                         dw_step=dw_step,
                         sample_module=self.sample_moudle,
                         points_batch_size=cfg['render_points_batch_size'], 
                         ray_batch_size= cfg['render_ray_batch_size'] )

        #===========================================================
        # check loss 

        self.map_loss_cands=[
                'color', 
                'depth',
                'obj_depth',
                'obj_color',
                'color_L2', 
                'depth_L2',
                'depth_freespace_weight',
                'objvec_L1',
                'objvec_L1_clamp',
                'objvec_CE_clamp',
                'objvec_CE',
                'objvec_BCE_clamp',
                'eikonal_L1',
                'eikonal_L2',
                'numerical_eikonal_L1',
                'sample_eikonal_L2',
                'mini_deform',
                'zero_deform',
                'deform_eikonal',
                'lipschitz_L1',
                'lipschitz_FT_L1',
                'lipschitz_FT_INF',
                'FT_eikonal_L2',
                'view_angle',
                'sf_normal',
                'sf_zero_sdf',
                'sf_sdf_normal_v3',
            ]

        self.track_loss_cands=[ 'sdf_L2', 
                                'rgb_L2' ]

        def loss_wt_factory(cfg,  guard_cand_list):
            
            _sett = {}
            for ln in guard_cand_list:
                _sett[ln]=None

            for lname,lwt in cfg.loss :
                assert lname in guard_cand_list, lname
                _sett[lname]=lwt 

            loss_weights =SimpleNamespace(**_sett)

            return loss_weights

        self.loss_wt_factory = loss_wt_factory

        #===========================================================

        # tracking_loss_cands=[
        #         'sdf_L2', 
        #         'rgb_L2',
        #         'tracker_ft_bce',
        #     ]

        # track_sett = {}
        # for ln in tracking_loss_cands:
        #     track_sett[ln]=None

        # self.tracking_loss_names=[]
        # for lname,lwt in self.shared_cfg.tracking_loss :
        #     assert lname in tracking_loss_cands
        #     track_sett[lname]=lwt 
        #
        #     self.tracking_loss_names.append(lname)
        #
        # self.tracking_loss_weights =SimpleNamespace(**track_sett)
        #
        # print(self.tracking_loss_names)
        # print(self.tracking_loss_weights)

        #===========================================================

        if args.add_ts:
            from time import gmtime, strftime
            ts = strftime("%m_%d_%H%M_%S", gmtime())

            self.output = self.output+'_'+ts 

        self.ckptsdir = os.path.join(self.output, 'ckpts')
        self.vis_sdir = f'{self.output}/vis'
        self.tb_sdir = f'{self.output}/tb'

        # parser.add_argument('--no_dir', action='store_true', default=False )
        if not args.cont and not args.debug and args.mode == 'train':
            assert not os.path.exists(self.vis_sdir), 'output directory already exists:\n'+self.vis_sdir

        self.reload_name = args.cont 
        clog.info(f'output: {self.output}'   ) 
        clog.info(f'ckpts:  {self.ckptsdir}' )
        clog.info(f'vis:    {self.vis_sdir}' )

        if self.reload_name is None and not args.no_dir:
            os.makedirs(self.output,   exist_ok=True) 
            os.makedirs(self.ckptsdir, exist_ok=True)
            os.makedirs(self.tb_sdir,  exist_ok=True)
            os.makedirs(f'{self.vis_sdir}', exist_ok=True)  
    


    #======================================================== 
    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)

        self.H, self.W   = self.cfg['cam']['H'], self.cfg['cam']['W']
        self.fx, self.fy = self.cfg['cam']['fx'], self.cfg['cam']['fy']
        self.cx, self.cy = self.cfg['cam']['cx'], self.cfg['cam']['cy']

        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H  -= self.cfg['cam']['crop_edge']*2
            self.W  -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']
    
    def set_lr(self, opt, decay, min_lr):
        
        for k in range(len(opt.param_groups)):

            old_lr = opt.param_groups[k]["lr"] 

            if old_lr != 0.0:
                new_lr = old_lr*decay 
                new_lr = max(min_lr,new_lr)
                opt.param_groups[k]["lr"] = new_lr
                print(f'[map][adj_lr][{k}]={old_lr*decay:.1e}')

    def set_lr_by_idxs(self, opt, decay, min_lr, idxs):
        
        # for k in range(len(opt.param_groups)):
        for i in idxs:

            old_lr = opt.param_groups[i]["lr"] 
                        
            if old_lr != 0.0:
                new_lr = old_lr*decay 
                new_lr = max(min_lr,new_lr)

                opt.param_groups[i]["lr"] = new_lr
                print(f'[adj_lr][{i}]={old_lr*decay:.1e}')

    def reset_lr_by_idxs(self, opt, new_lr, idxs):
        
        # for k in range(len(opt.param_groups)):
        for i in idxs:
            opt.param_groups[i]["lr"] = new_lr
            print(f'[adj_lr][{i}]={new_lr:.1e}')

    #==========================================================================
    def tx_sce_data(self, sce_models, device):
        
        for i in range(len(sce_models)):
            for tk in sce_models[i]:
                if not isinstance(sce_models[i][tk],float):
                    sce_models[i][tk] = sce_models[i][tk].to(device)

        return sce_models

        # sce_data_list=[]
        # for sce in sce_models:
        #     c  = sce['coarse_model'].clone_data(device)
        #     f  = sce['fine_model'].clone_data(device) 
        #     # order matter
        #     sce_data_list.append((c,f)) 
        # return sce_data 
    
    #=================================================================
    def save(self, save_fp,  network, obj_kf_poses, sce_models, step=None):


        obj_kf_poses_sd_list=[]

        for k in range(len(obj_kf_poses)):
            sd = obj_kf_poses[k].state_dict()
            obj_kf_poses_sd_list.append(sd)

        save_data=[]
        for s in sce_models:
            save_data.append(
                {'state_dict':s['model'].state_dict(),
                 'min_bound':s['min_bound'],
                 'grid_size':s['grid_size'],
                })

        dt={ 'obj_kf_poses_sd_list':obj_kf_poses_sd_list,
             'sce_data':save_data,
             'network':self.network.state_dict(),
        }

        if step is not None:
            dt['iteration']=step

        torch.save( dt, save_fp )

        print('save at:\n'+save_fp)

    def load(self, load_fp, network, obj_kf_poses, sce_models ):
        
        #load_fp = os.path.join(self.ckptsdir,f'{prefix}.pch')
        print('[load]:'+load_fp)
        x = torch.load(load_fp)
        obj_kf_poses_sd_list = x['obj_kf_poses_sd_list']
        sce_data   = x['sce_data']
        network_sd = x['network']

        if 'iteration' in x:
            sstep = x['iteration']
        else:
            sstep = 0

        del x 

        # load pose
        for k in range(len(obj_kf_poses)):
            sd = obj_kf_poses_sd_list[k]
            obj_kf_poses[k].load_state_dict(sd)

        # load network
        network.load_state_dict(network_sd)

        # load sce
        # self.update_sce_models(sce_models, obj_sce_data_list)
        
        for i in range(len(sce_data)):
            new_sd = sce_data[i]['state_dict']
            sce_models[i]['model'].load_state_dict(new_sd)
            sce_models[i]['min_bound']=sce_data[i]['min_bound']
            sce_models[i]['grid_size']=sce_data[i]['grid_size'] 
            
        return sstep

    #=================================================================
    def f2f_save(self, prefix,  network, obj_kf_poses, sce_models, f_step):

        save_fp = os.path.join(self.ckptsdir,f'{prefix}.pch')

        obj_kf_poses_sd_list=[]

        for k in range(len(obj_kf_poses)):
            sd = obj_kf_poses[k].state_dict()
            obj_kf_poses_sd_list.append(sd)

        save_data=[]
        for s in sce_models:
            save_data.append(
                {'state_dict':s['model'].state_dict(),
                 'min_bound':s['min_bound'],
                 'grid_size':s['grid_size'],
                })

        dt={ 'obj_kf_poses_sd_list':obj_kf_poses_sd_list,
             'sce_data':save_data,
             'network':self.network.state_dict(), 
             'f_step':f_step,
        }

        torch.save( dt, save_fp )

        print('save at:\n'+save_fp)

    def f2f_load(self, load_fp, network, obj_kf_poses, sce_models ):
        
        assert os.path.exists(load_fp), load_fp

        x = torch.load(load_fp)
        obj_kf_poses_sd_list = x['obj_kf_poses_sd_list']
        sce_data   = x['sce_data']
        network_sd = x['network']
        f_step=x['f_step']
        # it = x['iteration']
        del x 

        # load pose
        for k in range(len(obj_kf_poses)):
            sd = obj_kf_poses_sd_list[k]
            obj_kf_poses[k].load_state_dict(sd)

        # load network
        network.load_state_dict(network_sd)

        # load sce
        # self.update_sce_models(sce_models, obj_sce_data_list)
        
        for i in range(len(sce_data)):
            new_sd = sce_data[i]['state_dict']
            sce_models[i]['model'].load_state_dict(new_sd)
            sce_models[i]['min_bound']=sce_data[i]['min_bound']
            sce_models[i]['grid_size']=sce_data[i]['grid_size']

        return f_step

    #=============================================================== 
    def opt_vis_obj(self, sub_save_dir, prefix, opt_cfg, keyframes, keyframes_indices, sce_data, bounds_world, kf_poses, idx2tidx, obj_uid, no_mp4, step=None):

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        scfg = self.shared_cfg
        near_z = scfg.near_z
        far_z  = scfg.far_z


        #-------------------------------------------------  
        cmap =plt.get_cmap('viridis')
        cmap2=plt.get_cmap('plasma')
        write_2_tb = getattr(self,'tbwt',None) is not None and step is not None

        assert isinstance(keyframes,list)

        k_img_list=[]  

        all_dep_err=[]
        all_rgb_err=[]
        for i in tqdm(range(len(keyframes))) :

            idx = keyframes_indices[i]

            ds=self.renderer.dw_step

            gt_depth_np_full = keyframes[i]['depth'].clone().detach().cpu()
            gt_depth_np_full = gt_depth_np_full.numpy()

            gt_color_np_full = keyframes[i]['color'].clone().detach().cpu()
            gt_color_np_full = gt_color_np_full.numpy()

            gt_seg_mask = keyframes[i]['seg'] == obj_uid
            gt_seg_mask = gt_seg_mask.detach().cpu().numpy()  
            gt_seg_mask = gt_seg_mask.astype(np.float32)
            gt_seg_mask = cv2.resize(gt_seg_mask,
                                (W//ds,H//ds),cv2.INTER_NEAREST)
            gt_seg_mask = gt_seg_mask>0

                        
            i_mask = self.get_obj_mask(
                        mode=opt_cfg.collect_sample_mask_mode, 
                        frame=keyframes[i], uid=obj_uid, device='cpu')

            i_mask  = i_mask.detach().cpu().numpy()  

            assert i_mask.sum()>0
            i_mask=i_mask.astype(np.float32)
            i_mask=cv2.resize(i_mask,(W//ds,H//ds),cv2.INTER_NEAREST)
            i_mask=i_mask>0 
                  
            gt_depth_np=cv2.resize(gt_depth_np_full,(W//ds,H//ds),cv2.INTER_NEAREST)

            gt_color_np=cv2.resize(gt_color_np_full,(W//ds,H//ds),cv2.INTER_CUBIC)


            gt_color_np[~i_mask]=0
            gt_depth_np[~i_mask]=0  

            #-------------------------------------------------

            rt = self.renderer.render_img( 
                            obj_sce_data_list=[ sce_data ],
                            obj_bounds_world=bounds_world, 
                            obj_idx2tidx=[idx2tidx],
                            f_idx=idx ,
                            decoder=self.network,  
                            obj_kf_poses=[ kf_poses ],
                            device=self.device,
                            near_z=near_z,
                            far_z =far_z, 
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None,)

            pred_depth = rt['depth']
            pred_color = rt['color']

            if self.render_shading:
                pred_shading = rt['shading']
                pred_shading_np = pred_shading.detach().cpu().numpy() 
            else:
                pred_shading_np = None

            pred_depth_np = pred_depth.detach().cpu().numpy()
            pred_color_np = pred_color.detach().cpu().numpy() 
            
            save_fp = os.path.join(sub_save_dir,
                            f'{prefix}_obj{obj_uid}_f{idx:03d}.png' )
            
            vis_geo_and_save_render_mask( 
                         gt_color_np, gt_depth_np, 
                         pred_color_np, pred_depth_np, pred_shading_np,
                         save_fp, 
                         mask=gt_seg_mask)

            if write_2_tb:  
                self.tbwt.add_image(mode='train',
                                    group_name=f'{prefix}vis_obj',
                                    name=f'{prefix}pred_color_f{idx:03d}_obj{obj_uid}',
                                    value=pred_color_np,
                                    dataformats='HWC',
                                    step=step)

                # _im=axs[0, 2].imshow(depth_residual, cmap="viridis", vmin=0, vmax=0.04)
                rgb_res = np.abs(pred_color_np-gt_color_np) 
                rgb_err = np.mean(rgb_res[gt_seg_mask])
                all_rgb_err.append(rgb_err)

                if self.render_shading:
                    _name = f'{prefix}pred_shading_f{idx:03d}_obj{obj_uid}'
                    self.tbwt.add_image(mode='train',
                                        group_name=f'{prefix}vis_obj',
                                        name=_name,
                                        value=pred_shading_np,
                                        dataformats='HWC',
                                        step=step)

                pred_depth_np_nv = (pred_depth_np-near_z)/(far_z-near_z)
                vis_pred_depth_np = cmap2(pred_depth_np_nv)[:,:,:3]

                _name = f'{prefix}pred_depth_f{idx:03d}_obj{obj_uid}'
                self.tbwt.add_image(mode='train',
                                    group_name=f'{prefix}vis_obj',
                                    name=_name,
                                    value=vis_pred_depth_np,
                                    dataformats='HWC',
                                    step=step)

                dres = np.abs(pred_depth_np-gt_depth_np)
                dres[~gt_seg_mask]=0.0
                dres[gt_depth_np==0]=0.0

                d_err = np.mean(dres[gt_seg_mask*(gt_depth_np>0)])
                all_dep_err.append(d_err)

                for th in [0.1]:                
                    dres_nv = (dres)/th
                    dres_nv = np.clip(dres_nv,0.0,1.0)
                    dres_img = cmap(dres_nv) 
                    
                    _name = f'{prefix}depth_residuals_f{idx:03d}_obj{obj_uid}/th{th:.2f}'

                    # image 
                    self.tbwt.add_image(
                            mode='train',
                            group_name=f'{prefix}vis_obj_err',
                            name=_name,
                            value=dres_img,
                            dataformats='HWC',
                            step=step)


            k_img_list.append(save_fp)
            gc.collect()

        #-------------------------------------------------
        if write_2_tb:

            mean_dep_err = np.mean(all_dep_err)
            mean_rgb_err = np.mean(all_rgb_err)
            # number 
            self.tbwt.add_scalar(
                    mode='train', 
                    group_name=f'{prefix}eval',
                    name=f'{prefix}eval/depth_residuals/obj_{obj_uid}',
                    value=mean_dep_err, 
                    step=step) 

            self.tbwt.add_scalar(
                    mode='train', 
                    group_name=f'{prefix}eval',
                    name=f'{prefix}eval/color_residuals/obj_{obj_uid}',
                    value=mean_rgb_err, 
                    step=step) 

        #-------------------------------------------------
        # save mp4 
        if len(k_img_list)>2 and not no_mp4:
            save_fp = os.path.join(sub_save_dir,f'video_{prefix}.mp4' ) 
            make_video(k_img_list, save_fp, fps=3)


    def opt_vis_all(self, sub_save_dir, prefix, keyframes, keyframes_indices, obj_sce_data_list, obj_bounds_world, obj_kf_poses, obj_idx2tidx, no_mp4, step=None):

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        scfg = self.shared_cfg
        near_z = scfg.near_z
        far_z  = scfg.far_z

        #-----------------------------------------------------------------
        pbar = tqdm(range(len(keyframes)))
        cmap =plt.get_cmap('viridis')
        cmap2=plt.get_cmap('plasma')
        write_2_tb = getattr(self,'tbwt',None) is not None and step is not None

        assert isinstance(keyframes,list)
        assert len(keyframes)>0

        all_dep_err=[]
        all_rgb_err=[]

        img_list =[]

        for i in pbar:
            idx = keyframes_indices[i] 

            gt_depth_np_full = keyframes[i]['depth'].detach().cpu().numpy()
            gt_color_np_full = keyframes[i]['color'].detach().cpu().numpy() 
            
            #-------------------------------------------------      
            ds=self.renderer.dw_step

            gt_depth_np=cv2.resize(gt_depth_np_full,(W//ds,H//ds),cv2.INTER_NEAREST)
            gt_color_np=cv2.resize(gt_color_np_full,(W//ds,H//ds),cv2.INTER_CUBIC)
            #-------------------------------------------------      
            rt= self.renderer.render_img(
                            obj_sce_data_list=obj_sce_data_list,
                            obj_bounds_world=obj_bounds_world, 
                            obj_idx2tidx=obj_idx2tidx,
                            f_idx=idx,
                            decoder=self.network,  
                            obj_kf_poses=obj_kf_poses,
                            device=self.device,
                            near_z=near_z,
                            far_z =far_z,
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None )

            pred_depth = rt['depth']
            pred_color = rt['color']

            if self.render_shading: 
                pred_shading = rt['shading']
                pred_shading_np = pred_shading.detach().cpu().numpy() 
            else:
                pred_shading_np =None

            #-----------------------------------------------------
            pred_depth_np   = pred_depth.detach().cpu().numpy()
            pred_color_np   = pred_color.detach().cpu().numpy() 
            
            save_fp = os.path.join(sub_save_dir,f'f{idx:03d}_{prefix}.png' )  

            vis_geo_and_save_render_mask( 
                         gt_color_np, gt_depth_np, 
                         pred_color_np, pred_depth_np, pred_shading_np,
                         save_fp, 
                         mask=None)

            img_list.append(save_fp)
            #-----------------------------------------------------  
            if write_2_tb:  
                self.tbwt.add_image(mode='train',
                                    group_name=f'{prefix}vis_scene',
                                    name=f'{prefix}pred_color_f{idx:03d}_all',
                                    value=pred_color_np,
                                    dataformats='HWC',
                                    step=step)

                rgb_res = np.abs(pred_color_np-gt_color_np) 
                rgb_err = np.mean(rgb_res)
                all_rgb_err.append(rgb_err) 

                if self.render_shading: 
                    self.tbwt.add_image(mode='train',
                                        group_name=f'{prefix}vis_scene',
                                        name=f'{prefix}pred_shading_f{idx:03d}_all',
                                        value=pred_shading_np,
                                        dataformats='HWC',
                                        step=step)

                pred_depth_np_nv = (pred_depth_np-near_z)/(far_z-near_z)
                pred_depth_np_nv = np.clip(pred_depth_np_nv,0,1)
                pred_depth_np_vis = cmap2(pred_depth_np_nv)[:,:,:3]
                self.tbwt.add_image(mode='train',
                                    group_name=f'{prefix}vis_scene',
                                    name=f'{prefix}pred_depth_f{idx:03d}_all',
                                    value=pred_depth_np_vis,
                                    dataformats='HWC',
                                    step=step)

                dres = np.abs(pred_depth_np-gt_depth_np) 
                dres[gt_depth_np==0]=0.0

                d_err = np.mean(dres[(gt_depth_np>0)]) 
                all_dep_err.append(d_err)

                # for th in [0.02, 0.04,0.1]:                
                for th in [0.1]:                
                    dres_nv = (dres)/th
                    dres_nv = np.clip(dres_nv,0.0,1.0)
                    dres_img = cmap(dres_nv) 
                    
                    self.tbwt.add_image(
                            mode='train',
                            group_name=f'{prefix}vis_scene_err',
                            name=f'{prefix}depth_residuals/th{th:.2f}/f{idx:03d}',
                            value=dres_img,
                            dataformats='HWC',
                            step=step)
            
            gc.collect()

        #------------------------------------------------
        if write_2_tb:

            mean_dep_err = np.mean(all_dep_err)
            mean_rgb_err = np.mean(all_rgb_err)

            self.tbwt.add_scalar(
                    mode='train', 
                    group_name=f'{prefix}eval_scene',
                    name=f'{prefix}eval_scene/depth_residuals',
                    value=mean_dep_err, 
                    step=step) 

            self.tbwt.add_scalar(
                    mode='train', 
                    group_name=f'{prefix}eval_scene',
                    name=f'{prefix}eval_scene/color_residuals',
                    value=mean_rgb_err, 
                    step=step) 

        #--------------------------------------------------
        # save mp4 
        if len(img_list)>2 and not no_mp4:
            save_fp = os.path.join(sub_save_dir,f'video_{prefix}.mp4' ) 
            make_video(img_list, save_fp, fps=3)

    def opt_vis_pose(self, sub_save_dir, prefix, cur_fidx, obj_kf_poses, obj_uids, obj_idx2tidx, step=None, max_len=None, post_fix=''):

        scfg = self.shared_cfg
        cmap=plt.get_cmap('viridis')
        write_2_tb = getattr(self,'tbwt',None) is not None and step is not None

        k_fidxs = self.opt_frames_indices

        for k in range(len(obj_uids)):

            if self.pose_vis_lims[k] is None:
                k_lim1,k_lim2=None,None
            else:
                k_lim1,k_lim2=self.pose_vis_lims[k]

            k_pose  = obj_kf_poses[k]
            k_uid   = obj_uids[k]  

            est_poses = k_pose.get_all_rigid_poses()
            est_poses = [e.cpu().detach().numpy() for e in est_poses]
            
            gt_poses  = self.obj_gt_poses[k_uid]

            ini_poses = self.obj_ini_poses[k_uid]
            #--------------------------------------------
            save_fp_bar = os.path.join(sub_save_dir, 
                    f'{prefix}_pose_errbar_obj_{k_uid}_{prefix}_{step}.jpg')


            #--------------------------------------------
            if cur_fidx is None:
                fig_bar = plot_pose_errbar_v2(est_poses, 
                                              pose_ini=ini_poses, 
                                              pose_gt=gt_poses, 
                                              save_fp=save_fp_bar, 
                                              return_fig=True,
                                              cut=True)

            else:
                fig_bar = plot_pose_errbar_v2(est_poses[:cur_fidx+1], 
                                            pose_ini=ini_poses[:cur_fidx+1], 
                                            pose_gt=gt_poses[:cur_fidx+1], 
                                            save_fp=save_fp_bar, 
                                            return_fig=True,
                                            cut=True)

            if write_2_tb:
                self.tbwt.add_figure(
                                mode='train', 
                                group_name=f'{prefix}pose', 
                                name=f'{prefix}pose_errbar/obj_{k_uid}{post_fix}', 
                                value=fig_bar, 
                                step=step )

            #--------------------------------------------
            save_fp=os.path.join(sub_save_dir,f'{prefix}_pose_obj_{k_uid}_{prefix}_{step}.jpg')

            if cur_fidx is None:  
                _est_poses = np.stack(est_poses,axis=0) 
                _gt_poses = np.stack(gt_poses,axis=0) 

                _lim1,_lim2,fig=plot_draw_poses(
                                pose=_est_poses, 
                                pose_ref=_gt_poses, 
                                save_fp=save_fp, 
                                return_fig=True,
                                lim1=None,
                                lim2=None) 

            elif max_len is not None and cur_fidx is not None:
                s=max(0,cur_fidx-max_len+1)

                _est_poses = np.stack(est_poses,axis=0)[s:cur_fidx+1]
                _gt_poses = np.stack(gt_poses,axis=0)[s:cur_fidx+1]

                _lim1,_lim2,fig=plot_draw_poses(
                                pose=_est_poses, 
                                pose_ref=_gt_poses, 
                                save_fp=save_fp, 
                                return_fig=True,
                                lim1=None,
                                lim2=None) 

            else:
                _est_poses = np.stack(est_poses,axis=0)[:cur_fidx+1]
                _gt_poses = np.stack(gt_poses,axis=0)[:cur_fidx+1]

                _lim1,_lim2,fig=plot_draw_poses(
                                pose=_est_poses, 
                                pose_ref=_gt_poses, 
                                save_fp=save_fp, 
                                return_fig=True,
                                lim1=k_lim1,
                                lim2=k_lim2)

                if self.pose_vis_lims[k] is None:
                    self.pose_vis_lims[k]=(_lim1,_lim2)

            if write_2_tb:
                self.tbwt.add_figure(
                                mode='train', 
                                group_name=f'{prefix}pose', 
                                name=f'{prefix}pose/obj_{k_uid}{post_fix}', 
                                value=fig, 
                                step=step )


            plt.close(fig)

        plt.close('all')
        

    def render_all(self, sub_save_dir, prefix, keyframes, keyframes_indices, obj_sce_data_list, obj_bounds_world, obj_kf_poses, obj_idx2tidx  ):

        scfg = self.shared_cfg
        near_z = scfg.near_z
        far_z  = scfg.far_z

        #----------------------------
        pbar = tqdm(range(len(keyframes)))

        assert len(keyframes)>0

        gt_depth_list  =[]
        gt_rgb_list    =[]
        gt_seg_list    =[]

        pred_depth_list=[]
        pred_rgb_list  =[]
        pred_shading_list=[]

        indices=[]
        img_list=[]
        

        for i in pbar:
            gt_depth_np = keyframes[i]['depth'].detach().cpu().numpy()
            gt_color_np = keyframes[i]['color'].detach().cpu().numpy() 
            gt_seg_np = keyframes[i]['seg'].detach().cpu().numpy() 
            
            idx = keyframes_indices[i] 
            indices.append(idx)

            # save_sh_fp = os.path.join(sub_save_dir,
            #                 f'sh_f{idx:03d}{prefix}.png' ) 

            #-----------------------------------------------------
            rt = self.renderer.render_img( obj_sce_data_list=obj_sce_data_list,
                            obj_bounds_world=obj_bounds_world, 
                            obj_idx2tidx=obj_idx2tidx,
                            f_idx=idx,
                            decoder=self.network,  
                            obj_kf_poses=obj_kf_poses,
                            device=self.device,
                            near_z=near_z,
                            far_z =far_z, 
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None )

            pred_depth = rt['depth']
            pred_color = rt['color']

            if self.render_shading:
                pred_shading = rt['shading']
                pred_shading_np = pred_shading.detach().cpu().numpy()
                pred_shading_list.append(pred_shading_np)

            #-----------------------------------------------------
            pred_depth_np = pred_depth.detach().cpu().numpy()
            pred_depth_list.append(pred_depth_np)
            gt_depth_list.append(gt_depth_np)
            
            pred_color_np = pred_color.detach().cpu().numpy()  
            pred_rgb_list.append(pred_color_np)
            gt_rgb_list.append(gt_color_np)           


            gt_seg_list.append(gt_seg_np)

            # #-----------------------------------------------------
            # if save_img:
            #     _color_np= pred_color_np*255
            #     _color_np= _color_np.astype(np.uint8)

            #     _sh_np = pred_shading_np*255
            #     _sh_np= _sh_np.astype(np.uint8)

            #     _vis = np.hstack([_color_np,_sh_np])

            #     skimage.io.imsave(save_fp, _vis)

            #     img_list.append(save_fp)

            #----------------------------------------------------- 
            gc.collect()

        #---------------------------------------------------------- 

        rt={
            'indices':indices,
            'pred_rgb_list':pred_rgb_list,
            'gt_rgb_list'  :gt_rgb_list,
            'pred_depth_list':pred_depth_list,
            'gt_depth_list':gt_depth_list,
            'gt_seg_list':gt_seg_list,
            'pred_shading_list':pred_shading_list,
        }

        return rt

    def eval_render(self, rt, txt_fp, target_uid, eval_masked_pred, debug_save_dir ):
        #
        # rt={
        #     'pred_color_np':pred_color_np,
        #     'pred_depth_np':pred_depth_np,
        #     'gt_rgb_list'  :gt_rgb_list,
        #     'gt_depth_list':gt_depth_list,
        # }
        # 
        indices=rt['indices']
        pred_rgb_list=rt['pred_rgb_list']
        gt_rgb_list  =rt['gt_rgb_list']
        gt_seg_list  =rt['gt_seg_list']

        all_metrics={
            'indices':[],
            'psnr':[],
            'ssim':[],
            'lpips':[],
        }

        fnum = len(pred_rgb_list)

        masked_gt_list =[]

        for i in range(fnum):

            _idx = indices[i]
            pred = pred_rgb_list[i].copy()
            gt   = gt_rgb_list[i].copy()
            seg  = gt_seg_list[i].copy()

            if target_uid is not None:

                mask = np.zeros_like(seg,dtype=np.bool_)
                mask[(seg==target_uid)]=1

                gt[~mask]=0

                if eval_masked_pred:
                    pred[~mask]=0

                fp1 = os.path.join(debug_save_dir,f'debug_pred_{target_uid}_{_idx:03d}.png')
                fp2 = os.path.join(debug_save_dir,f'debug_gt_{target_uid}_{_idx:03d}.png')
                #fp3 = os.path.join(v,f'debug_mask_{_idx:03d}.png')

                pred_save = (pred*255).astype(np.uint8)
                gt_save   = (gt*255).astype(np.uint8)
                #mask_vis  = (mask*255).astype(np.uint8) 
                #mask_vis = np.stack([mask_vis,mask_vis,mask_vis],axis=-1)

                if i==0:
                    skimage.io.imsave(fp1,pred_save)
                    skimage.io.imsave(fp2,gt_save)
                #skimage.io.imsave(fp3,mask_vis) 
                

            i_psnr = src.metrics.eval_psnr(np_gt=gt,np_im=pred)
            i_ssim = src.metrics.eval_ssim(np_gt=gt,np_im=pred,max_val=1)
            i_lpips= src.metrics.eval_lpips(np_gt=gt,np_im=pred,net_name='vgg',device=self.device)

            all_metrics['indices'].append(_idx)
            all_metrics['psnr'].append(i_psnr)
            all_metrics['ssim'].append(i_ssim)
            all_metrics['lpips'].append(i_lpips)

        if txt_fp is not None:
            with open(txt_fp,'w')as fout:

                for tag in ['psnr','ssim','lpips']:
                    fout.write(f'{tag}\t')

                fout.write('\n')

                for i in range(fnum):
                    for tag in ['psnr','ssim','lpips']:
                        val = all_metrics[tag][i]
                        fout.write(f'{val:.4f}\t')

                    fout.write('\n')

                fout.write('\n')
                fout.write('mean\n')
                for tag in ['psnr','ssim','lpips']:
                    fout.write(f'{tag}\t')
                fout.write('\n')

                # mean
                for tag in ['psnr','ssim','lpips']:
                    mean_val = np.mean(all_metrics[tag])
                    fout.write(f'{mean_val:.4f}\t')

                fout.write('\n')

                fout.write('\n')  

        print('eval. done')
        return all_metrics

    def handle_loss_log(self, loss_log, name_prefix):  
        steps= loss_log['step']

        for k,v in loss_log.items():
            
            if len(v)>0 and k!='step': 

                for ii in range(len(steps)):
                    st  = steps[ii]
                    val = v[ii]
                    if val is None:
                        continue

                    if '[par]' in k:
                        self.tbwt.add_scalar(
                                mode='train', 
                                group_name=f'{name_prefix}params',
                                name=f'{name_prefix}params/{k}',
                                value=val, 
                                step=st) 

                    else:
                        self.tbwt.add_scalar(
                                mode='train', 
                                group_name=f'{name_prefix}loss_{k}',
                                name=f'{name_prefix}loss',
                                value=val, 
                                step=st) 

        for k in loss_log.keys():
            loss_log[k]=[]

    def eval_train_poses(self, obj_kf_poses, obj_uids, tag, accum_step):

        scfg = self.shared_cfg

        for k,k_uid in enumerate(obj_uids):

            # numpy 
            k_est_poses = obj_kf_poses[k].get_all_rigid_poses()
            k_est_poses = [e.cpu().detach().numpy() for e in k_est_poses]
            k_gt_poses  = self.obj_gt_poses[k_uid]

            rot_err =[]
            trsl_err=[]

            for ix,e in enumerate(k_est_poses):
                i_rot_err = (e[:3,:3]-k_gt_poses[ix][:3,:3])
                i_rot_err = np.abs(i_rot_err).mean()
                rot_err.append(i_rot_err)
                
                i_trsl_err = (e[:3,3]-k_gt_poses[ix][:3,3])
                i_trsl_err = np.abs(i_trsl_err).mean()
                trsl_err.append(i_trsl_err)

            rot_err = np.mean(rot_err)
            trsl_err = np.mean(trsl_err) 

            self.tbwt.add_scalar(
                mode='train', 
                group_name=f'{tag}_train_eval',
                name=f'{tag}_train_eval/obj_{k_uid}/rot_err',
                value=rot_err, 
                step=accum_step) 

            self.tbwt.add_scalar(
                mode='train', 
                group_name=f'{tag}_train_eval',
                name=f'{tag}_train_eval/obj_{k_uid}/trsl_err',
                value=trsl_err, 
                step=accum_step) 


    #============================================================== 
    def get_obj_mask(self , mode, frame , uid , device):

        if mode =='v2_seg': 
            seg = frame['seg']
            obj_mask = (seg == uid)

        elif mode =='v3_objs_mask': 
            obj_mask = frame['objs_mask'][uid] 

        elif mode=='v4_bg_full_v3_objs_mask':
            if uid==0: # BG
                obj_mask = torch.ones_like(gt_depth, device=device, dtype=torch.bool)
            else:
                obj_mask = frame['objs_mask'][uid]  

        elif mode =='v5_objs_ex_mask': 
            obj_mask = frame['objs_ex_mask'][uid] 

        else:
            raise Exception('error mode:'+mode)

        return obj_mask.to(device)


    def collect_samples_cam_s(self, mask_mode, optimize_frames, rand_sample_num, sel_obj_uids):
        #
        #
        # F,N

        device = self.device
        scfg   = self.shared_cfg

        H, W, fx, fy = self.H, self.W, self.fx, self.fy, 
        cx, cy = self.cx, self.cy

        bH, eH = 0, H
        bW, eW = 0, W 

        batch_rays_d_list   = []
        batch_rays_o_list   = []
        batch_gt_depth_list = []
        batch_gt_color_list = []
        batch_gt_seg_list   = []
        batch_gt_tkft_list  = []
        batch_gt_nv_list  = []
        batch_im_iw_list    = []
        batch_im_ih_list    = []

        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H  

        sample_pix_num = rand_sample_num//len(optimize_frames)
        sample_pix_num = sample_pix_num//len(sel_obj_uids)

        for frame in optimize_frames:
            gt_depth = frame['depth'].clone().detach()
            gt_color = frame['color'].clone().detach()
            gt_seg   = frame['seg'].clone().detach()
            gt_nv    = frame['normal'].clone().detach() 
            pose     = torch.eye(4, device=device) 
            
            obj_rays_o_list   = []
            obj_rays_d_list   = []

            obj_gt_depth_list = []
            obj_gt_color_list = []
            obj_gt_seg_list = []
            obj_gt_tkft_list = []
            obj_gt_nv_list=[]

            obj_iw_list=[]
            obj_ih_list=[]

            for k in range(len(sel_obj_uids)):                
                uid = sel_obj_uids[k]

                obj_mask = self.get_obj_mask(mask_mode, frame, 
                                        uid=uid, device=device)

                #--------------------------------------                
                if scfg.train_sample_alg=='v3_sample_mask': 

                    obj_depth = gt_depth.clone().detach()
                    obj_depth[~obj_mask]=0

                    rt = get_samples_v3_my( sample_pix_num, 
                                            H, W, fx, fy, cx, cy, pose, 
                                            obj_depth, gt_color, gt_seg, device)

                    _rays_o, _rays_d, _gt_depth, _gt_color, _gt_seg, _iw, _ih= rt 

                elif scfg.train_sample_alg=='v4_sample_mask_and_boundary':
                    
                    obj_depth = gt_depth.clone().detach() 

                    p_mask2 = (gt_seg==uid).to(device) # obj
                    p_mask1 = obj_mask * (p_mask2==0)  # freespace
                    
                    subsample2=int(sample_pix_num*0.5)
                    subsample1=sample_pix_num-subsample2

                    rt = get_samples_v4_my( subsample1, subsample2, 
                                            H, W, fx, fy, cx, cy, pose, 
                                            obj_depth, 
                                            gt_color,
                                            p_mask1=p_mask1,
                                            p_mask2=p_mask2, 
                                            seg=gt_seg, 
                                            device=device)
                    

                    _rays_o, _rays_d, _gt_depth, _gt_color, _gt_seg,_iw,_ih = rt 

                elif scfg.train_sample_alg=='v6_balanced_obj_freespace':
                    
                    obj_depth = gt_depth.clone().detach() 

                    p_mask2 = (gt_seg==uid).to(device) # obj
                    p_mask1 = ~p_mask2 # freespace
                    
                    subsample2=int(sample_pix_num*0.5)
                    subsample1=sample_pix_num-subsample2

                    rt = get_samples_v4_my( subsample1, subsample2, 
                                            H, W, fx, fy, cx, cy, pose, 
                                            obj_depth, 
                                            gt_color,
                                            p_mask1=p_mask1,
                                            p_mask2=p_mask2, 
                                            seg=gt_seg, 
                                            device=device)
                    

                    _rays_o, _rays_d, _gt_depth, _gt_color, _gt_seg,_iw,_ih = rt 
                else:
                    raise Exception('err:'+  scfg.train_sample_alg)

                # n,c
                #_rays_tkft=tkft[_ih,_iw]
                
                _rays_nv = gt_nv[_ih,_iw]

                obj_rays_o_list.append(_rays_o)
                obj_rays_d_list.append(_rays_d)
                obj_gt_depth_list.append(_gt_depth)
                obj_gt_color_list.append(_gt_color)
                obj_gt_seg_list.append(_gt_seg)
                obj_gt_nv_list.append(_rays_nv)
                #obj_gt_tkft_list.append(_rays_tkft)

                obj_iw_list.append(_iw)
                obj_ih_list.append(_ih)

            obj_rays_o   = torch.cat(obj_rays_o_list,dim=0)
            obj_rays_d   = torch.cat(obj_rays_d_list,dim=0)
            obj_gt_depth = torch.cat(obj_gt_depth_list,dim=0)
            obj_gt_color = torch.cat(obj_gt_color_list,dim=0)
            obj_gt_seg   = torch.cat(obj_gt_seg_list,dim=0)
            obj_gt_nv    = torch.cat(obj_gt_nv_list,dim=0)
            #obj_gt_tkft  = torch.cat(obj_gt_tkft_list,dim=0)
            obj_iw   = torch.cat(obj_iw_list,dim=0)
            obj_ih   = torch.cat(obj_ih_list,dim=0)


            batch_rays_o_list.append(obj_rays_o)
            batch_rays_d_list.append(obj_rays_d)
            
            batch_gt_depth_list.append(obj_gt_depth)
            batch_gt_color_list.append(obj_gt_color)
            batch_gt_seg_list.append(obj_gt_seg)
            #batch_gt_tkft_list.append(obj_gt_tkft)
            batch_gt_nv_list.append(obj_gt_nv) 

            batch_im_iw_list.append(obj_iw)
            batch_im_ih_list.append(obj_ih)


        batch_rays_o = torch.stack(batch_rays_o_list,0)

        # ray points on the image plane (depth = 1)
        batch_rays_d = torch.stack(batch_rays_d_list,0)

        batch_gt_depth = torch.stack(batch_gt_depth_list,0)
        batch_gt_color = torch.stack(batch_gt_color_list,0)
        batch_gt_seg   = torch.stack(batch_gt_seg_list,0) 
        batch_gt_nv = torch.stack(batch_gt_nv_list,0)
        batch_im_iw = torch.stack(batch_im_iw_list,0)
        batch_im_ih = torch.stack(batch_im_ih_list,0)

        rays={
            'batch_rays_o':batch_rays_o,
            'batch_rays_d':batch_rays_d,
            'batch_gt_depth':batch_gt_depth,
            'batch_gt_color':batch_gt_color,
            'batch_gt_seg':batch_gt_seg,
            'batch_gt_nv':batch_gt_nv,
            'batch_im_iw':batch_im_iw,
            'batch_im_ih':batch_im_ih,
        }

        return rays 


    def sample_frame(self, keyframes, keyframes_indices, rand_sample_num, sel_obj_uids, obj_idx2tidx):
        #
        # F,N

        device = self.device
        scfg   = self.shared_cfg

        H, W, fx, fy = self.H, self.W, self.fx, self.fy, 
        cx, cy = self.cx, self.cy

        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H  

        sample_pix_num = rand_sample_num//len(keyframes)
        sample_pix_num = sample_pix_num//len(sel_obj_uids)

        obj_samples=defaultdict(list)

        for k in range(len(sel_obj_uids)):                
            uid = sel_obj_uids[k]

            frame_samples=defaultdict(list)
            k_idx2tidx = obj_idx2tidx[k] 

            for j,frame in enumerate(keyframes):
                gt_depth = frame['depth'].clone().detach()
                gt_color = frame['color'].clone().detach()
                #gt_tkft  = frame['tracker_ft'].clone().detach()
                gt_nv    = frame['normal'].clone().detach()
                gt_seg   = frame['seg'].clone().detach()
                pose     = torch.eye(4, device=device) 

                p_mask = uid==gt_seg

                #  subsample1,  H, W, fx, fy, cx, cy, c2w, depth, color, p_mask, seg, device
                rt = get_surface_sample_v2( 
                                    rand_sample_num,
                                    H, W, fx, fy, cx, cy, 
                                    c2w=pose, 
                                    depth=gt_depth,
                                    color=gt_color, 
                                    p_mask=p_mask, 
                                    seg=gt_seg,
                                    device=device)

                # rt['sample_pts']=sf_pts
                # rt['sample_color']=sample_color
                # rt['sample_depth']=sample_depth
                # rt['sample_seg']=sample_seg
                # rt['iw']=iw
                # rt['ih']=ih 
                for tk,val in rt.items():
                    frame_samples[tk].append(val) 


                t = k_idx2tidx[keyframes_indices[j]]
                t_idx = torch.tensor(t).reshape(1,)
                #t_idx = repeat(t_idx, ' 1 -> n ', n=rt['sample_pts'].shape[0])
                frame_samples['sample_tidx'].append(t_idx)


            for tk,val in frame_samples.items():
                obj_samples[tk].append( torch.stack(val,0) )


        # k,f,n,c
        for tk,val in obj_samples.items():
            obj_samples[tk] = torch.stack(val,0) 

        # rt['sample_pts']=sf_pts
        # rt['sample_color']=sample_color
        # rt['sample_depth']=sample_depth
        # rt['sample_seg']=sample_seg
        # rt['iw']=iw
        # rt['ih']=ih 
        return obj_samples

    #============================================================
    def opt_mulobj_helper(self, opt_cfg, lswt, cam_rays, f_indices, obj_kf_poses, obj_uids, obj_bounds_world, obj_sce_data_list, network, obj_idx2t, max_pre_samples, device, step=None):

        device = self.device     
        H, W,  = self.H, self.W
        scfg   = self.shared_cfg

        near_z = scfg.near_z
        far_z  = scfg.far_z 

        #--------------------------------- 

        render_obj_vec=False
        render_obj_vec+=lswt.objvec_L1 is not None 
        render_obj_vec+=lswt.objvec_L1_clamp is not None 
        render_obj_vec+=lswt.objvec_BCE_clamp is not None 
        render_obj_vec+=lswt.objvec_CE_clamp is not None 
        render_obj_vec+=lswt.objvec_CE is not None 

        # render_obj_vec+=lswt.objvec_BCE   is not None 
        # render_obj_vec+=lswt.objvec_CE    is not None 
        # render_obj_vec+=lswt.fs_objvec_L1 is not None
        # render_tkft = lswt.tracker_ft is not None

        esti_normal = lswt.sample_eikonal_L2

        out = self.sample_moudle.render_rays_w_objs(
                        cam_rays_o=cam_rays['batch_rays_o'], 
                        cam_rays_d=cam_rays['batch_rays_d'],
                        f_indices=f_indices,
                        obj_idx2tidx=obj_idx2t,
                        obj_kf_poses=obj_kf_poses,
                        obj_bounds_world=obj_bounds_world,
                        obj_sce_data_list=obj_sce_data_list,
                        decoder=network,
                        render_sample_mode=scfg.ray_sample_mode,
                        render_sample_num=scfg.ray_sample_num,
                        near_z=near_z, 
                        far_z =far_z, 
                        device=device, 
                        esti_normal=esti_normal,
                        render_obj_vec=render_obj_vec,
                        #render_tracker_ft=False,
                        max_pre_samples=max_pre_samples,
                        verbose=False
                    )

        debug_train_pts=out['debug_train_pts'].clone().detach().cpu().numpy()
        debug_valid    =out['debug_valid'].clone().detach().cpu().numpy()

        debug={
            'debug_train_pts':debug_train_pts,
            'debug_valid':debug_valid,
        }
        #---------------------------------------------
        pred_depth = out['depth']
        pred_color = out['color']

        # f n k m c
        pred_shift  = out['p_shift']
        # f n k m
        pred_shift_valid  = out['p_shift_valid']


        if esti_normal:
            # f n k m c
            pred_p_grad = out['gradient']
            pred_p_grad_val = out['grad_val']

        if lswt.sf_sdf_normal_v3 is not None:
            # k f n 
            cano_obj_rays_valid = out['cano_obj_rays_valid']
            cano_obj_rays_o     = out['cano_obj_rays_o']
            cano_obj_rays_d     = out['cano_obj_rays_d']

        #---------------------------------------------
        ray_gt_depth  = cam_rays['batch_gt_depth']
        ray_gt_rgb    = cam_rays['batch_gt_color']
        ray_gt_seg    = cam_rays['batch_gt_seg']
        # ray_gt_nv     = cam_rays['batch_gt_nv']
        # ray_gt_tkft   = cam_rays['batch_gt_tkft']

        ray_isobj = None 
        for k_obj_uid in obj_uids:
            if ray_isobj is not None:
                ray_isobj = ray_isobj + (ray_gt_seg==k_obj_uid)
            else:
                ray_isobj = (ray_gt_seg==k_obj_uid)

        #---------------------------------------------
        # k f n m
        obj_sample_z = out['obj_sample_z']
        # k f n m
        obj_sample_fs_mask = torch.zeros_like(obj_sample_z).to(torch.bool) 

        _gt_depth = ray_gt_depth.unsqueeze(-1)

        for k in range(obj_sample_z.shape[0]):

            # f n m
            om = (obj_sample_z[k]<_gt_depth)
            obj_sample_fs_mask[k,om]=1

        #--------------------------------------------- 
        losses={}

        #---------------------------------------------
        if lswt.obj_color is not None:
            assert ray_gt_rgb.shape == pred_color.shape 
            color_err  = torch.abs(ray_gt_rgb - pred_color)

            obj_color_err = color_err[ray_isobj>0] 
            obj_color_loss = obj_color_err.mean() 
            losses['obj_color'] = obj_color_loss*lswt.obj_color

        #---------------------------------------------
        if lswt.obj_depth is not None: 
            assert ray_gt_depth.shape == pred_depth.shape 

            dep_err_l1 = torch.abs(ray_gt_depth-pred_depth)
            
            val_mask =  (ray_gt_depth > 0) * ray_isobj>0
            
            assert val_mask.sum()>0

            obj_depth_loss = (dep_err_l1[val_mask]).mean() 

            losses['obj_depth'] = obj_depth_loss*lswt.obj_depth

        #---------------------------------------------
        if lswt.color is not None:
            assert ray_gt_rgb.shape == pred_color.shape 
            color_err  = torch.abs(ray_gt_rgb - pred_color)
            
            color_loss = color_err.mean() 
            losses['color'] = color_loss*lswt.color

        #---------------------------------------------
        if lswt.color_L2 is not None:
            assert ray_gt_rgb.shape == pred_color.shape 
            color_err  = (ray_gt_rgb - pred_color)**2
            color_loss = color_err.mean() 
            losses['color_L2'] = color_loss*lswt.color

        #---------------------------------------------
        if lswt.depth is not None:
            assert ray_gt_depth.shape == pred_depth.shape 

            dep_err_l1 = torch.abs(ray_gt_depth-pred_depth)
            
            val_mask =  (ray_gt_depth > 0)  
            
            assert val_mask.sum()>0

            depth_loss = (dep_err_l1[val_mask]).mean() 

            losses['depth'] = depth_loss*lswt.depth

        if lswt.depth_L2  is not None:
            assert ray_gt_depth.shape == pred_depth.shape 

            dep_err = (ray_gt_depth-pred_depth)**2
            
            val_mask =  (ray_gt_depth > 0)  
            
            assert val_mask.sum()>0

            depth_L2_loss = (dep_err[val_mask]).mean() 

            losses['depth_L2'] = depth_L2_loss*lswt.depth_L2

        #---------------------------------------------
        if lswt.objvec_L1 is not None:
            # f n k
            pred_objvec = out['objvec'] 
            pred_objvec = rearrange(pred_objvec,'f n k -> k f n') 
            
            #k f n
            gt_label = torch.zeros_like(pred_objvec) 
            
            # make gt
            for k,uid in enumerate(obj_uids):
                m = ray_gt_seg==uid # f n 
                gt_label[k,m]=1 

            ce_l1_err  = (gt_label-pred_objvec).abs()
            ce_l1_loss = (ce_l1_err.mean())
            
            losses['objvec_L1'] = ce_l1_loss* lswt.objvec_L1
  
        #---------------------------------------------
        if lswt.objvec_L1_clamp is not None:
            # f n k
            pred_objvec = out['objvec'] 
            pred_objvec = rearrange(pred_objvec,'f n k -> k f n') 
            
            # k f n
            gt_label = torch.zeros_like(pred_objvec) 
            
            # make gt
            for k,uid in enumerate(obj_uids):
                m = ray_gt_seg==uid # f n 
                gt_label[k,m]=1 

            pred_objvec = pred_objvec.clip(1e-3, 1.0-1e-3)
            ce_l1_err  = (gt_label-pred_objvec).abs()
            ce_l1_loss = (ce_l1_err.mean())
            
            losses['objvec_L1_clamp'] = ce_l1_loss* lswt.objvec_L1_clamp


        #---------------------------------------------
        if lswt.objvec_BCE_clamp is not None:
            # f n k
            pred_objvec = out['objvec'] 
            # k f n
            pred_objvec = rearrange(pred_objvec,'f n k -> k f n') 

            pred_objvec = pred_objvec.clip(1e-3, 1.0 - 1e-3) 
            
            # f n
            bce_err_list=[]
            for k,uid in enumerate(obj_uids):
                
                # f n
                k_pred = pred_objvec[k]
                
                k_gt_y = (ray_gt_seg==uid)
                
                k_gt_y = k_gt_y.to(torch.float32)

                bce_err = torch.nn.functional.binary_cross_entropy(
                                            input=k_pred, 
                                            target=k_gt_y,
                                            reduction='mean')
                
                bce_err_list.append(bce_err) 
                
            bce_loss = torch.stack(bce_err_list).mean()
            losses['objvec_BCE_clamp'] = bce_loss* lswt.objvec_BCE_clamp
            assert bce_loss.requires_grad
            
            # mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        #---------------------------------------------
        if lswt.objvec_CE_clamp is not None:

            pred_objvec = out['objvec'] 
            pred_objvec = rearrange(pred_objvec,'f n k -> f k n') 

            # f n
            # initial = index 0
            gt_label = torch.zeros_like(ray_gt_seg).to(torch.long)

            assert len(obj_uids)>1, 'must have two objects'
            
            for k,uid in enumerate(obj_uids):
                m = ray_gt_seg==uid
                gt_label[m]=k

            pred_objvec = pred_objvec.clip(1e-3, 1.0 - 1e-3) 

            # N,C
            # input:  f,k,n
            # target: f,n  with value \in [0,C]
            ce_err = torch.nn.functional.cross_entropy(
                                            input=pred_objvec, 
                                            target=gt_label, 
                                            reduction='none')

            ce_loss = (ce_err.mean()) 

            losses['objvec_CE_clamp']=ce_loss*lswt.objvec_CE_clamp
            assert ce_loss.requires_grad

        if lswt.objvec_CE is not None:

            # 1 0 0 * w_1
            # 0 1 0 * w_2
            # 1 0 0 * w_3
            # +=
            # 1 1 0
            pred_objvec = out['objvec'] 
            pred_objvec = rearrange(pred_objvec,'f n k -> f k n') 

            #pred_objvec = pred_objvec.clip(1e-3, 1.0 - 1e-3) 

            # f n
            # initial = index 0
            gt_label = torch.zeros_like(ray_gt_seg).to(torch.long)

            assert len(obj_uids)>1, 'must have two objects'
            
            for k,uid in enumerate(obj_uids):
                m = ray_gt_seg==uid
                gt_label[m]=k

            # N,C
            # input:  f,k,n
            # target: f,n  with value \in [0,C]
            ce_err = torch.nn.functional.cross_entropy(
                                            input=pred_objvec, 
                                            target=gt_label, 
                                            reduction='none')

            ce_loss = (ce_err.mean()) 

            losses['objvec_CE']=ce_loss*lswt.objvec_CE
            assert ce_loss.requires_grad

            pdb.set_trace()
        #---------------------------------------------
        if lswt.mini_deform is not None:
            if pred_shift_valid.sum()>0:
                mini_deform_loss = pred_shift[pred_shift_valid].abs().mean() 
                
                mini_deform_loss = mini_deform_loss

                losses['mini_deform'] = mini_deform_loss*lswt.mini_deform

        #---------------------------------------------
        if lswt.zero_deform is not None:
            fs_mask = rearrange(obj_sample_fs_mask,'k f n m -> f n k m')

            # f n k m
            # freespace_mask = pred_shift_valid.clone().detach()
            
            # f n k m
            #freespace_mask[gt_exist]=0
        
            if fs_mask.sum()>0:
                # f n k m
                abs_delta = pred_shift[fs_mask].abs()
                zero_deform_loss = abs_delta.mean()
        
                losses['zero_deform'] = zero_deform_loss*lswt.zero_deform
            else:
                losses['zero_deform'] = None 

        if lswt.depth_freespace_weight:
            # k f n m-1
            obj_weights  = out['obj_weights']
            # k f n m
            obj_sample_z = out['obj_sample_z']

            # k f n m
            # obj_sample_fs_mask

            # k f n m-1
            fs_sample_mask = obj_sample_fs_mask[...,:-1]

            if fs_sample_mask.sum()>0:
                socc_err= (obj_weights[fs_sample_mask]).abs().mean() 
                _wt = lswt.depth_freespace_weight
                losses['depth_freespace_weight']=socc_err*_wt
            else:
                losses['depth_freespace_weight']=None 

        #---------------------------------------------
        if lswt.deform_eikonal is not None:

            #snum=32
            snum=opt_cfg.eikonal_sample_num

            df_ek_loss=[]
            for k in range(len(obj_sce_data_list)):
                random_p = torch.rand(snum,3)*2-1
                random_p = random_p.to(device)
                random_p.requires_grad=True 

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(low=0, high=fnum,
                                           size=(random_p.shape[0],)) 

                rt = network(random_p, 
                                  scene_data=obj_sce_data_list[k], 
                                  motion_net=k_obj_pose, 
                                  p_tidx=rand_p_tidx, 
                                  normalize=False,
                                  deform=(not k_obj_pose.is_rigid()),
                                  ) 

                ek_sdf  = rt['sdf']
                _tx_noc_p  = rt['_tx_noc_p']
                ek_grad = grad.gradient(ek_sdf, _tx_noc_p)

                k_ek_err = (ek_grad.norm(dim=-1)-1) 
                k_ek_err = (k_ek_err**2).mean()
                df_ek_loss.append(k_ek_err) 
            
            deform_eikonal_loss = torch.mean(*df_ek_loss)

            losses['deform_eikonal'] = deform_eikonal_loss*lswt.deform_eikonal

        
        if lswt.eikonal_L1 is not None :

            snum=opt_cfg.eikonal_sample_num
            eikonal_loss=[]
            for k in range(len(obj_sce_data_list)):

                random_p = torch.rand(snum,3)*2-1
                random_p = random_p.to(device)
                random_p.requires_grad=True 

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p.shape[0],)) 

                rt = network(random_p, 
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                x_sdf  = rt['sdf']
                x_grad = grad.gradient(x_sdf, random_p)

                x_grad_nm = torch.linalg.norm(x_grad,  dim=-1)
                k_ek_err = (x_grad_nm-1.0).abs().mean()

                eikonal_loss.append(k_ek_err)

            eikonal_loss = torch.stack(eikonal_loss,dim=0).mean()
            # eikonal_loss =  eikonal_loss*( 1.0/len(obj_sce_data_list) )

            losses['eikonal_L1'] = eikonal_loss*lswt.eikonal_L1


        if lswt.eikonal_L2 is not None :

            snum=opt_cfg.eikonal_sample_num
            eikonal_loss_L2=[]

            for k in range(len(obj_sce_data_list)):
                random_p = torch.rand(snum,3)*2-1
                random_p = random_p.to(device)
                random_p.requires_grad=True 

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p.shape[0],)) 


                # def forward(self, p, vdirs, scene_data, normalize, motion_net, p_tidx, deform=False ):
                rt = network(random_p,
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                x_sdf  = rt['sdf']
                x_grad = grad.gradient(x_sdf, random_p)

                x_grad_nm = torch.linalg.norm(x_grad,  dim=-1)
                k_ek_sqerr = (x_grad_nm-1.0)**2
                k_ek_sqerr = k_ek_sqerr.mean()

                eikonal_loss_L2.append(k_ek_sqerr)

            eikonal_loss_L2 = torch.stack(eikonal_loss_L2,dim=0).mean()

            losses['eikonal_L2'] = eikonal_loss_L2*lswt.eikonal_L2


        if lswt.sample_eikonal_L2 is not None : 

            obj_p_grad  = rearrange(pred_p_grad,'f n k m c -> k m f n c')
            obj_p_valid = rearrange(pred_p_grad_val,'f n k m -> k m f n')

            _v_grad  = obj_p_grad[obj_p_valid]
            _grad_nm = torch.linalg.norm(_v_grad, dim=-1)
            ek_sqerr = (_grad_nm-1.0)**2

            sample_ek_l2 = ek_sqerr.mean()

            losses['sample_eikonal_L2'] = sample_ek_l2*lswt.sample_eikonal_L2

        
        #---------------------------------------------
        if lswt.numerical_eikonal_L1 is not None :

            snum=opt_cfg.eikonal_sample_num
            ek_loss_list=[]
            for k in range(len(obj_sce_data_list)):

                random_p = torch.rand(snum,3)*2-1
                random_p = random_p.to(device) 

                max_delta=0.05
                ns = (torch.randn(snum,3)*max_delta).to(device) 
                too_small=ns.abs()<1.0e-4 
                ns[too_small]=0.0001
                random_p2 = random_p + ns
                random_p2 = random_p2.clamp(-1,1)
                random_p2 = random_p2.to(device) 

                # n,3
                delta=(random_p2-random_p)

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p.shape[0],)) 

                rt = network(random_p, 
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                x_sdf1  = rt['sdf']

                rt2 = network(random_p2, 
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                x_sdf2  = rt2['sdf'] 

                #--------------------------------------
                # n,3
                x_grad = (x_sdf2 - x_sdf1) / delta
                x_grad_nm = torch.linalg.norm(x_grad,  dim=-1)
                k_ek_err = (x_grad_nm-1.0).abs().mean()

                ek_loss_list.append( k_ek_err )

            ek_loss2=torch.stack(ek_loss_list,dim=0).mean()

            losses['numerical_eikonal_L1'] = ek_loss2*lswt.numerical_eikonal_L1

        #---------------------------------------------
        if lswt.lipschitz_L1 is not None:
            
            snum =scfg.lipschitz_sample_num
            delta=scfg.lipschitz_step_size

            lipz_err_list=[]

            for k in range(len(obj_sce_data_list)):

                random_p1 = torch.rand(snum,3)*2-1
                random_p1 = random_p1.to(device)

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p1.shape[0],)) 

                offset = torch.rand_like(random_p1)*2-1
                offset = offset*delta

                random_p2 = random_p1+offset

                # random_p1.requires_grad=True 
                # random_p2.requires_grad=True 

                rt1 = network( random_p1, 
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                rt2 = network( random_p2, 
                              vdirs=None,
                              scene_data=obj_sce_data_list[k], 
                              motion_net=k_obj_pose, 
                              p_tidx=rand_p_tidx, 
                              normalize=False,
                              deform=False,
                              sdf_only=True) 

                x_diff = (random_p1-random_p1).abs().sum(dim=-1)
                y_diff = (rt1['sdf']- rt2['sdf']).abs().sum(dim=-1)
                
                # y_diff <= x_diff * c
                mask = y_diff > x_diff

                if mask.sum()>0:
                    lipz_err = y_diff[mask].abs().mean()
                    lipz_err_list.append(lipz_err)


            if len(lipz_err_list)>0:
                lipz_loss = torch.stack(lipz_err_list,dim=0).mean()
                losses['lipschitz_L1'] = lipz_loss*lswt.lipschitz_L1
            else:
                losses['lipschitz_L1'] = None 
        
        #---------------------------------------------
        if lswt.lipschitz_FT_L1 is not None:
            
            snum =scfg.lipschitz_sample_num
            delta=scfg.lipschitz_step_size

            lipzFT_err_list=[]

            for k in range(len(obj_sce_data_list)):

                random_p1 = torch.rand(snum,3)*2-1
                random_p1 = random_p1.to(device)

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p1.shape[0],)) 

                offset = torch.rand_like(random_p1)*2-1
                offset = offset*delta

                random_p2 = random_p1+offset

                rt1 = network.forward_enc( random_p1, 
                                           vdirs=None,
                                           scene_data=obj_sce_data_list[k], 
                                           normalize=False,
                                           motion_net=k_obj_pose, 
                                           p_tidx=rand_p_tidx, ) 

                rt2 = network.forward_enc( random_p2, 
                                           vdirs=None,
                                           scene_data=obj_sce_data_list[k], 
                                           normalize=False,
                                           motion_net=k_obj_pose, 
                                           p_tidx=rand_p_tidx, ) 

                dd = (rt1['sdf_ft'] - rt2['sdf_ft'])
                x_diff = dd.abs().sum(dim=-1)

                assert rt1['sdf'].shape[-1]==1
                y_diff = (rt1['sdf']- rt2['sdf']).abs().sum(dim=-1)

                #lipzFT_err = (x_diff-y_diff).abs().mean()
                #lipzFT_err_list.append(lipzFT_err) 

                # y_diff <= x_diff * c
                mask = y_diff > x_diff

                if mask.sum()>0:
                    lipzFT_err = y_diff[mask].abs().mean()
                    lipzFT_err_list.append(lipzFT_err) 

            #-----------------------------
            if len(lipzFT_err_list)>0:
                lipzFT_loss = torch.stack(lipzFT_err_list,dim=0).mean()
                losses['lipschitz_FT_L1'] = lipzFT_loss*lswt.lipschitz_FT_L1
            else:
                losses['lipschitz_FT_L1'] = None 
        
        #-------------------------------------------------------    
        if lswt.lipschitz_FT_INF is not None:
            
            snum =scfg.lipschitz_sample_num
            delta=scfg.lipschitz_step_size

            lipzFT_INF_err_list=[]

            for k in range(len(obj_sce_data_list)):

                random_p1 = torch.rand(snum,3)*2-1
                random_p1 = random_p1.to(device)

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p1.shape[0],)) 

                offset = torch.rand_like(random_p1)*2-1
                offset = offset*delta

                random_p2 = random_p1+offset

                rt1 = network.forward_enc( random_p1, 
                                           vdirs=None,
                                           scene_data=obj_sce_data_list[k], 
                                           normalize=False,
                                           motion_net=k_obj_pose, 
                                           p_tidx=rand_p_tidx, ) 

                rt2 = network.forward_enc( random_p2, 
                                           vdirs=None,
                                           scene_data=obj_sce_data_list[k], 
                                           normalize=False,
                                           motion_net=k_obj_pose, 
                                           p_tidx=rand_p_tidx, ) 

                xdd = (rt1['sdf_ft'] - rt2['sdf_ft']) 
                x_diff = xdd.abs().max(dim=-1)[0]

                assert rt1['sdf'].shape[-1]==1

                ydd = (rt1['sdf']- rt2['sdf'])
                #y_diff = torch.linalg.norm(ydd, dim=-1,ord='inf')
                y_diff = ydd.abs().max(dim=-1)[0]

                # y_diff <= x_diff * c
                mask = y_diff > x_diff

                if mask.sum()>0:
                    lipzFT_INF_err = y_diff[mask].abs().mean()
                    lipzFT_INF_err_list.append(lipzFT_INF_err) 

            #-----------------------------
            if len(lipzFT_INF_err_list)>0:
                lipzFT_INF_loss = torch.stack(lipzFT_INF_err_list,dim=0).mean()
                losses['lipschitz_FT_INF'] = lipzFT_INF_loss*lswt.lipschitz_FT_INF
            else:
                losses['lipschitz_FT_INF'] = None 
        
        #-------------------------------------------------------    
        if lswt.FT_eikonal_L2 is not None:
            
            snum =scfg.lipschitz_sample_num
            delta=scfg.lipschitz_step_size

            FT_eikonal_list=[]

            for k in range(len(obj_sce_data_list)):

                random_p1 = torch.rand(snum,3)*2-1
                random_p1 = random_p1.to(device)

                k_obj_pose = obj_kf_poses[k]
                fnum = k_obj_pose.get_frame_num()

                rand_p_tidx= torch.randint(
                                low=0, high=fnum,
                                size=(random_p1.shape[0],)) 


                rt1 = network.forward(random_p1, 
                                      vdirs=None,
                                      scene_data=obj_sce_data_list[k], 
                                      motion_net=k_obj_pose, 
                                      p_tidx=rand_p_tidx, 
                                      normalize=False,
                                      deform=False,
                                      sdf_only=True) 

                xx = rt1['p_h']
                grad_nm = torch.linalg.norm(xx, dim=-1)
                _sqerr = (grad_nm-1.0)**2

                _err = _sqerr.abs().mean()
                FT_eikonal_list.append(_sqerr) 

            #-----------------------------
            if len(FT_eikonal_list)>0:
                FT_eikonal_loss = torch.stack(FT_eikonal_list,dim=0).mean()
                losses['FT_eikonal_L2'] = FT_eikonal_loss*lswt.FT_eikonal_L2
            else:
                losses['FT_eikonal_L2'] = None 
        
        #-------------------------------------------------------
        if lswt.view_angle is not None:

            cal_vn_loss = lswt.view_normal is not None

            # f n c 
            cam_o = cam_rays['batch_rays_o']
            cam_d = cam_rays['batch_rays_d']
            cam_z = cam_rays['batch_gt_depth'] 

            # f n c
            cam_vdirs   = cam_o + cam_d
            cam_vdirs   = torch.nn.functional.normalize(cam_vdirs,dim=-1)
            #cam_vdirs  = cam_vdirs.reshape(-1,3)

            # f n c
            cam_sf_pts = cam_o + cam_d*(cam_z.unsqueeze(-1))
            #cam_sf_pts = cam_sf_pts.reshape(-1,3) 

            vn_errs=[]

            for k in range(len(obj_sce_data_list)):

                # x,3
                m_vdirs= cam_vdirs.clone().reshape(-1,3) 
                m_pts  = cam_sf_pts.clone().reshape(-1,3) 

                m_vdirs= rearrange(m_vdirs, 'x c -> x c 1')
                m_pts  = rearrange(m_pts,   'x c -> x c 1')

                k_tt = []
                for ff in f_indices:
                    k_tt.append(obj_idx2t[k][ff])
                k_tt = torch.tensor(k_tt).to(device)
                k_tt = repeat(k_tt,'f -> f n', n=cam_sf_pts.shape[1])

                # x
                m_tt = k_tt.reshape(-1)

                k_pnet = obj_kf_poses[k]

                mat44 = k_pnet.get_all_rigid_poses()
                # x,4,4
                m_mat44 = mat44[m_tt]

                # x,3,3
                m_rot  = m_mat44[:,:3,:3]
                # x,3,1
                m_trsl = m_mat44[:,:3, 3:]

                # x,3,1
                m_pts2 = torch.bmm(m_rot, m_pts) + m_trsl
                # x,3
                m_pts2 = m_pts2.squeeze(-1)
                m_pts2 = m_pts2.detach()
                m_pts2.requires_grad=True

                # x,3
                m_vdirs2 = torch.bmm(m_rot, m_vdirs)
                m_vdirs2 = m_vdirs2.squeeze(-1)
                m_vdirs2 = m_vdirs2.detach()

                _o =  network( m_pts2,
                               vdirs=None,
                               scene_data=obj_sce_data_list[k], 
                               motion_net=k_pnet, 
                               p_tidx=m_tt,
                               normalize=True,
                               deform=(not k_pnet.is_rigid()),
                               sdf_only=True)

                pred_sf_sdf = _o['sdf']

                x_grad  = grad.gradient(pred_sf_sdf, m_pts2)
                x_nv    = torch.nn.functional.normalize(x_grad,dim=-1)

                assert x_nv.shape == m_vdirs2.shape 

                # same sign = negative dot product -> clamp to 0
                vis_err = ((x_nv*m_vdirs2).sum(dim=-1)).clamp(min=0, max=None)
                vis_err = vis_err.mean()

                vn_errs.append(vis_err)

            

            vn_loss = torch.stack(vn_errs,dim=0).mean()
            losses['view_angle'] = vn_loss*lswt.view_angle

        
        #-------------------------------------------------------
        if lswt.sf_sdf_normal_v3 is not None:

            # f n c 
            cam_o  = cam_rays['batch_rays_o']
            cam_d  = cam_rays['batch_rays_d']
            cam_z  = cam_rays['batch_gt_depth'] 
            cam_nv = cam_rays['batch_gt_nv'] 

            # f n c
            cam_sf_pts = cam_o + cam_d*(cam_z.unsqueeze(-1))
            #cam_sf_pts = cam_sf_pts.reshape(-1,3) 

            nv_pred_sdf=[]    
            nv_in_pts=[]  
            nv_gt_nv=[]          

            for k in range(len(obj_sce_data_list)):
                
                # f n 
                k_instr = cano_obj_rays_valid[k]
                k_pnet = obj_kf_poses[k]

                if k_instr.sum()==0:
                    continue 

                m_gt_nv_list=[]
                m_pts_list=[]

                with torch.no_grad():
                    for fi,ff in enumerate(f_indices):
                        tt=obj_idx2t[k][ff]

                        f_mat44 = k_pnet.get_rigid_pose(tt)
                        #f_mat44= all_mat44[tt]

                        f_rot  = f_mat44[:3,:3]
                        # 3,1
                        f_trsl = f_mat44[:3,3].unsqueeze(-1)

                        # rdnum=64
                        # if m_pts2.shape[0]>rdnum:
                        #     rr = torch.randperm(m_pts2.shape[0])[:rdnum]
                        #     m_pts2  = m_pts2[rr]
                        #     m_gt_nv = m_gt_nv[rr]
                        #     m_tt = m_tt[rr]

                        # 3,n
                        f_gt_nv = torch.matmul(f_rot, cam_nv[fi].T)
                        
                        # 3,n
                        f_pts2  = torch.matmul(f_rot, cam_sf_pts[fi].T)
                        f_pts2  = f_pts2 + f_trsl

                        m_gt_nv_list.append(f_gt_nv.T)
                        m_pts_list.append(f_pts2.T)


                    m_gt_nv = torch.stack(m_gt_nv_list,dim=0)
                    m_pts   = torch.stack(m_pts_list,dim=0) 

                    m_pts2  =m_pts[k_instr].reshape(-1,3) 
                    m_gt_nv2=m_gt_nv[k_instr].reshape(-1,3).detach()


                    k_tt = []
                    for ff in f_indices:
                        k_tt.append(obj_idx2t[k][ff])

                    k_tt = torch.tensor(k_tt).to(device)
                    k_tt = repeat(k_tt,'f -> f n', n=cam_sf_pts.shape[1])
                    k_tt = k_tt[k_instr]
                    m_tt = k_tt.reshape(-1)


                    m_pts2 = m_pts2.detach()
                    m_pts2.requires_grad=True

                #with record_function("my_loss.sf_sdf_normal_v3.network"):  
                # main bottleneck
                _o =  network( m_pts2,
                               vdirs=None,
                               scene_data=obj_sce_data_list[k], 
                               motion_net=k_pnet, 
                               p_tidx=m_tt,
                               normalize=True,
                               deform=(not k_pnet.is_rigid()),
                               sdf_only=True)

                pred_sf_sdf = _o['sdf']
                
                assert 'normal' not in _o

                nv_in_pts.append(m_pts2)
                nv_pred_sdf.append(pred_sf_sdf)
                nv_gt_nv.append(m_gt_nv2)

            
            if len(nv_pred_sdf)>0:
                nv_pred_sdf2=torch.cat(nv_pred_sdf,dim=0)
                #nv_in_pts2=torch.cat(nv_in_pts,dim=0)
                nv_gt_nv2=torch.cat(nv_gt_nv,dim=0)

                x_grad  = grad.gradient_list(nv_pred_sdf2, nv_in_pts)
                x_grad  = torch.cat(x_grad,dim=0) 

                assert x_grad.requires_grad
                x_nv = torch.nn.functional.normalize(x_grad,dim=-1)

                assert x_nv.shape == nv_gt_nv2.shape 

                vn_loss   = (x_nv-nv_gt_nv2).abs().mean()  
                zsdf_loss = (nv_pred_sdf2).abs().mean() 
            
                losses['sf_sdf_normal_v3'] = (vn_loss+zsdf_loss)*lswt.sf_sdf_normal_v3 
            else:
                losses['sf_sdf_normal_v3'] = None 
        
        #-------------------------------------------------------
        if lswt.sf_normal is not None:

            # f n c 
            cam_o  = cam_rays['batch_rays_o']
            cam_d  = cam_rays['batch_rays_d']
            cam_z  = cam_rays['batch_gt_depth'] 
            cam_nv = cam_rays['batch_gt_nv'] 

            # f n c
            cam_sf_pts = cam_o + cam_d*(cam_z.unsqueeze(-1))
            #cam_sf_pts = cam_sf_pts.reshape(-1,3) 

            nv_errs=[]

            for k in range(len(obj_sce_data_list)):

                m_cam_nv= cam_nv.clone().reshape(-1,3) 
                m_cam_nv= rearrange(m_cam_nv, 'x c -> x c 1')

                # x,3
                m_pts  = cam_sf_pts.clone().reshape(-1,3) 
                m_pts  = rearrange(m_pts,   'x c -> x c 1')

                k_tt = []
                for ff in f_indices:
                    k_tt.append(obj_idx2t[k][ff])
                k_tt = torch.tensor(k_tt).to(device)
                k_tt = repeat(k_tt,'f -> f n', n=cam_sf_pts.shape[1])

                # x
                m_tt = k_tt.reshape(-1)

                k_pnet = obj_kf_poses[k]

                mat44 = k_pnet.get_all_rigid_poses()
                # x,4,4
                m_mat44 = mat44[m_tt]

                # x,3,3
                m_rot  = m_mat44[:,:3,:3]
                # x,3,1
                m_trsl = m_mat44[:,:3, 3:]

                # x,3,1
                m_pts2 = torch.bmm(m_rot, m_pts) + m_trsl
                # x,3
                m_pts2 = m_pts2.squeeze(-1)
                m_pts2 = m_pts2.detach()
                m_pts2.requires_grad=True

                # x,3
                m_gt_nv = torch.bmm(m_rot, m_cam_nv)
                m_gt_nv = m_gt_nv.squeeze(-1)
                m_gt_nv = m_gt_nv.detach()

                _o =  network( m_pts2,
                               vdirs=None,
                               scene_data=obj_sce_data_list[k], 
                               motion_net=k_pnet, 
                               p_tidx=m_tt,
                               normalize=True,
                               deform=(not k_pnet.is_rigid()),
                               sdf_only=True)

                pred_sf_sdf = _o['sdf']

                if 'normal' in _o and _o['normal']  is not None:
                    x_nv = _o['normal'] 
                else:
                    x_grad  = grad.gradient(pred_sf_sdf, m_pts2)
                    assert x_grad.requires_grad
                    x_nv    = torch.nn.functional.normalize(x_grad,dim=-1)

                assert x_nv.shape == m_gt_nv.shape 

                nv_err = (x_nv-m_gt_nv).abs().mean()

                nv_errs.append(nv_err)
            
            vn_loss = torch.stack(nv_errs,dim=0).mean()
            losses['sf_normal'] = vn_loss*lswt.sf_normal

        #-------------------------------------------------------
        if lswt.sf_zero_sdf is not None:

            # f n c 
            cam_o  = cam_rays['batch_rays_o']
            cam_d  = cam_rays['batch_rays_d']
            cam_z  = cam_rays['batch_gt_depth'] 
            cam_nv = cam_rays['batch_gt_nv'] 

            # f n c
            cam_sf_pts = cam_o + cam_d*(cam_z.unsqueeze(-1))
            #cam_sf_pts = cam_sf_pts.reshape(-1,3) 

            zsdf_errs=[]

            for k in range(len(obj_sce_data_list)):

                m_cam_nv= cam_nv.clone().reshape(-1,3) 
                m_cam_nv= rearrange(m_cam_nv, 'x c -> x c 1')

                # x,3
                m_pts  = cam_sf_pts.clone().reshape(-1,3) 
                m_pts  = rearrange(m_pts,   'x c -> x c 1')

                k_tt = []
                for ff in f_indices:
                    k_tt.append(obj_idx2t[k][ff])
                k_tt = torch.tensor(k_tt).to(device)
                k_tt = repeat(k_tt,'f -> f n', n=cam_sf_pts.shape[1])

                # x
                m_tt = k_tt.reshape(-1)

                k_pnet = obj_kf_poses[k]

                mat44 = k_pnet.get_all_rigid_poses()
                # x,4,4
                m_mat44 = mat44[m_tt]

                # x,3,3
                m_rot  = m_mat44[:,:3,:3]
                # x,3,1
                m_trsl = m_mat44[:,:3, 3:]

                # x,3,1
                m_pts2 = torch.bmm(m_rot, m_pts) + m_trsl
                # x,3
                m_pts2 = m_pts2.squeeze(-1)
                m_pts2 = m_pts2.detach()
                m_pts2.requires_grad=True

                _o =  network( m_pts2,
                               vdirs=None,
                               scene_data=obj_sce_data_list[k], 
                               motion_net=k_pnet, 
                               p_tidx=m_tt,
                               normalize=True,
                               deform=(not k_pnet.is_rigid()),
                               sdf_only=True)

                pred_sf_sdf = _o['sdf']

                zsdf_err = (pred_sf_sdf).abs().mean()

                zsdf_errs.append(zsdf_err)
            
            zsdf_loss = torch.stack(zsdf_errs,dim=0).mean()
            losses['sf_zero_sdf'] = zsdf_loss*lswt.sf_zero_sdf

        return losses,debug


    def cal_win(self, it, delay_step, win_interval, win_min):

        if delay_step !=-1:
            if it > delay_step:
                _ratio = ((it-delay_step)/win_interval)
                w= np.clip(_ratio, win_min, 1.0)
            else:
                w =0.0
        else:
            # no delay
            _ratio = (it/win_interval)
            w= np.clip(_ratio, win_min, 1.0)

        return w 

        
    #============================================================
    def f2f_map(self, 
        start_iters,
        total_iters, sce_models, 
        obj_kf_poses, obj_bounds, obj_uids, obj_idx2t,
        opt_cfg, 
        new_opt_keyframes,  new_opt_keyf_indices, 
        old_opt_keyframes,  old_opt_keyf_indices, 
        vis_keyframes, vis_keyf_indices, tag, do_BA , 
        enable_vis_all=True, 
        enable_vis_obj=True):

        #torch.autograd.set_detect_anomaly(True)
        
        lswt = self.loss_wt_factory(opt_cfg, self.map_loss_cands)

        self.pose_vis_lims = [None for k in range(len(sce_models))]

        #-------------------------------------------------------
        scfg = self.shared_cfg
        device = self.device 

        sce_params=[] 

        for s in sce_models:
            sce_params+=s['model'].parameters()


        deform_params=[] 
        deform_nets=[]
        for k in range(len(obj_kf_poses)): 
            net = obj_kf_poses[k] 
            if not net.is_rigid():
                x = net.get_params_by_flag(deform=True) 
                deform_params+=x
                deform_nets.append(net)

        rigid_params=[]  
        for k in range(len(obj_kf_poses)): 
            net = obj_kf_poses[k] 
            x = net.get_params_by_flag( rigid=True )
            rigid_params+=x

        tcode_params=[]
        for k in range(len(obj_kf_poses)): 
            net = obj_kf_poses[k] 
            x = net.get_params_by_flag(tcode=True )
            tcode_params+=x

        if do_BA:
            optimizer = torch.optim.Adam([  
                        {'params':sce_params,                
                            'lr': opt_cfg.ini_sce_lr    },  # 0
                        {'params':self.network.parameters(), 
                            'lr': opt_cfg.ini_net_lr    },  # 1
                        {'params':deform_params,                
                            'lr': opt_cfg.ini_deform_lr    }, # 2
                        {'params':rigid_params,                
                            'lr': opt_cfg.ini_rigidpose_lr    }, # 3
                        {'params':tcode_params,                
                            'lr': opt_cfg.ini_tocde_lr    }, # 4
                    ])

            sce_pg_idx  = 0
            net_pg_idx  = 1
            deform_pg_idx = 2
            rigid_pg_idx  = 3
            tcode_pg_idx  = 4

        else:
            optimizer = torch.optim.Adam([  
                        {'params':sce_params,                
                            'lr': opt_cfg.ini_sce_lr    },  # 0
                        {'params':self.network.parameters(), 
                            'lr': opt_cfg.ini_net_lr    },  # 1
                        {'params':deform_params,                
                            'lr': opt_cfg.ini_deform_lr    }, # 2
                        {'params':tcode_params,                
                            'lr': opt_cfg.ini_tocde_lr    }, # 3
                    ])

            sce_pg_idx     = 0
            net_pg_idx     = 1
            deform_pg_idx  = 2
            tcode_pg_idx  = 3

        #--------------------------------------------------------------------
        LR_cache = {}
        for ii in range(len(optimizer.param_groups)):
            LR_cache[ii] = optimizer.param_groups[ii]['lr']

        #--------------------------------------------------------------------
        glog_net_pm_list = [ ]
        for pm in self.network.log_grad_module:
            pp = pm.named_parameters() 
            new_pp =[ (xn,x) for xn,x in pp if 'weight_g' in xn ]

            if len(new_pp)==0: 
                new_pp =[ (xn,x) for xn,x in pp if 'weight' in xn ]

            glog_net_pm_list+=new_pp

        glog_deform_pm_list = [ ] 
        for i,pm in enumerate(deform_params): 
            glog_deform_pm_list.append((f'deform_p_{i}',pm))
        
        glog_rigid_pm_list = []
        for i,pm in enumerate(rigid_params):
            glog_rigid_pm_list.append((f'rigid_pose_p_{i}',pm))


        glog_sdfnet_list = [ ]
        for sd in sce_models: 

            pp = sd['model'].named_parameters() 
            new_pp =[ (xn,x) for xn,x in pp if 'weight_g' in xn ]

            if len(new_pp)==0: 
                new_pp =[ (xn,x) for xn,x in pp if 'weight' in xn ]

            glog_sdfnet_list+=new_pp


        pm_groups=[
            ('network',    glog_net_pm_list),
            ('sdfnet',     glog_sdfnet_list),
            ('deform',     glog_deform_pm_list),
        ]

        if do_BA:
            pm_groups.append( ('rigid_poses', glog_rigid_pm_list) )

        log_num= len(pm_groups)
        grad_log= [ defaultdict(list) for i in range(log_num) ] 

        #=============================================================
        # loss log
        loss_log=OrderedDict()
        tk_list = ['total','step']

        tk_list += self.map_loss_cands
        tk_list += [ f'[par]_LR{j}' for j in range(len(optimizer.param_groups)) ]
        tk_list += [  '[par]_deform_win_a' , '[par]_deform_win_b']

        for tk in tk_list:
            loss_log[tk]=[]

        #===================================
        if opt_cfg.use_deform_win_a:
            self.network._deform_win_a=0.0
        else:
            self.network._deform_win_a=1.0

        if opt_cfg.use_deform_win_b:
            self.network._deform_win_b=0.0
        else:
            self.network._deform_win_b=1.0

        #===================================
        # run scene model opt.

        timer = MyTimer(verbose=scfg.debug_time)

        pbar = tqdm(range(start_iters, total_iters))

        early_stop_th = opt_cfg.early_stop_th

        accum_step=start_iters-1 

        for it in pbar:
            
            timer.tstart('preprocess')

            accum_step+=1

            #===============================
            if opt_cfg.use_middle_to_fine_training:

                if it < opt_cfg.middle_iters:
                    self.network.set_stage('middle') 
                elif it < opt_cfg.middle_iters+opt_cfg.fine_iters:
                    self.network.set_stage('fine') 
                else:
                    self.network.set_stage('color')

            #===============================

            i_keyframes = []
            i_keyf_indices = []

            if opt_cfg.rand_img_per_batch != -1 and len(new_opt_keyframes) > opt_cfg.rand_img_per_batch:

                ridx = torch.randperm(len(new_opt_keyframes))[:opt_cfg.rand_img_per_batch]
                ridx = ridx.tolist() 

                for x in ridx:
                    #fi = new_opt_keyf_indices[x]
                    i_keyframes.append(new_opt_keyframes[x])
                    i_keyf_indices.append(new_opt_keyf_indices[x])
            else:
                i_keyframes    = new_opt_keyframes
                i_keyf_indices = new_opt_keyf_indices 

            #===============================
            i_old_keyframes = []
            i_old_keyf_indices = []

            if len(old_opt_keyframes)>0 :
                if opt_cfg.rand_img_per_batch != -1 and  len(old_opt_keyframes) > opt_cfg.rand_img_per_batch:

                    ridx = torch.randperm(len(i_old_keyframes))[:opt_cfg.rand_img_per_batch]

                    ridx = ridx.tolist() 

                    for x in ridx:
                        i_old_keyframes.append(    old_opt_keyframes[x]    )
                        i_old_keyf_indices.append( old_opt_keyf_indices[x] )

                else:
                    i_old_keyframes    = old_opt_keyframes
                    i_old_keyf_indices = old_opt_keyf_indices 

            #===============================
            # reset win_a/win_b

            if opt_cfg.use_deform_win_a:

                _wa=self.cal_win(it=it,
                            delay_step=opt_cfg.deform_win_a_delay_step,
                            win_interval=opt_cfg.deform_win_a_interval,
                            win_min=opt_cfg.deform_win_a_min)
                self.network._deform_win_a=_wa

            
            if opt_cfg.use_deform_win_b :

                _wb=self.cal_win(it=it,
                            delay_step=opt_cfg.deform_win_b_delay_step,
                            win_interval=opt_cfg.deform_win_b_interval,
                            win_min=0.0)

                self.network._deform_win_b=_wb

            vis_wina=self.network._deform_win_a

            timer.tend()
            
            #===================================
            # camera space   
            # optimize model 
            # f,n,c
            timer.tstart('collect_samples')

            sel_rand_crays = self.collect_samples_cam_s(
                                    opt_cfg.collect_sample_mask_mode,
                                    i_keyframes, 
                                    opt_cfg.rand_sample_num,
                                    obj_uids)
            
            sel_keyf_idxs = i_keyf_indices
            
            timer.tend()
            timer.tstart('opt_mulobj_helper')

            rd_ls, ddebug = self.opt_mulobj_helper( 
                                opt_cfg=opt_cfg,
                                lswt=lswt,
                                cam_rays=sel_rand_crays, 
                                f_indices=sel_keyf_idxs,
                                obj_kf_poses=obj_kf_poses, 
                                obj_bounds_world=obj_bounds, 
                                obj_uids=obj_uids,
                                obj_sce_data_list=sce_models,
                                network=self.network,
                                obj_idx2t=obj_idx2t, 
                                max_pre_samples=scfg.max_pre_samples,
                                device=device,
                                step=it) 

            timer.tend()

            total_loss = 0.0
            for ln,lval in rd_ls.items(): 
                if lval is not None:
                    assert lval.requires_grad, f'[no gd. error] {ln}'
                    total_loss += lval

            if len(i_old_keyframes)>0 :  
                sel_rand_crays2 = self.collect_samples_cam_s(
                                        opt_cfg.collect_sample_mask_mode,
                                        i_old_keyframes, 
                                        opt_cfg.rand_sample_num,
                                        obj_uids)
                
                sel_keyf_idxs2 = i_old_keyf_indices
                
                rd_ls2, _ = self.opt_mulobj_helper( 
                                    opt_cfg=opt_cfg,
                                    lswt=lswt,
                                    cam_rays=sel_rand_crays2, 
                                    f_indices=sel_keyf_idxs2,
                                    obj_kf_poses=obj_kf_poses, 
                                    obj_bounds_world=obj_bounds, 
                                    obj_uids=obj_uids,
                                    obj_sce_data_list=sce_models,
                                    network=self.network,
                                    obj_idx2t=obj_idx2t, 
                                    max_pre_samples=scfg.max_pre_samples,
                                    device=device) 

                for ln2,lval2 in rd_ls2.items(): 
                    if lval2 is not None:
                        assert lval2.requires_grad, f'[no gd. error] {ln2}'
                        total_loss += lval2


            #--------------
            if torch.isnan(total_loss).any():
                print('detect NaN')
                pdb.set_trace()

            if torch.isinf(total_loss).any():
                print('detect INF')
                pdb.set_trace()

            timer.tstart('backward')

            optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            optimizer.step() 

            timer.tend()

            #-----------------------------
            timer.tstart('post-process')
            # log
            loss_log['step'].append(accum_step)
            loss_log['total'].append(total_loss.item())  
            
            for jj in range(len(optimizer.param_groups)):
                loss_log[f'[par]_LR{jj}'].append(
                        optimizer.param_groups[jj]["lr"])

            loss_log['[par]_deform_win_a'].append(self.network._deform_win_a) 
            loss_log['[par]_deform_win_b'].append(self.network._deform_win_b) 

            for ln,lval in rd_ls.items(): 
                if lval is not None:
                    loss_log[ln].append(lval.item())
                else:
                    loss_log[ln].append(None)

            #-----------------------------
            if it %4==0:
                pbar.set_description(f'[{tag}{it:4d}] win_a={vis_wina:.1e} loss={total_loss.item():.1e}')

            if it %100==0 and it >0:
                self.handle_loss_log(loss_log=loss_log, 
                            name_prefix=tag)

            if it in opt_cfg.sce_decay_lr_at: 
                self.set_lr_by_idxs(optimizer, opt_cfg.lr_decay_ratio, opt_cfg.min_lr,[sce_pg_idx])

            if it in opt_cfg.net_decay_lr_at:
                self.set_lr_by_idxs(optimizer, opt_cfg.lr_decay_ratio, opt_cfg.min_lr, [net_pg_idx] )

            if it in opt_cfg.deformnet_decay_lr_at:
                self.set_lr_by_idxs(optimizer, opt_cfg.lr_decay_ratio, opt_cfg.min_lr, [deform_pg_idx] )

            if it in opt_cfg.rigidpose_decay_lr_at:
                self.set_lr_by_idxs(optimizer, opt_cfg.lr_decay_ratio, opt_cfg.min_lr, [rigid_pg_idx] )

            if it in opt_cfg.tcode_decay_lr_at:
                self.set_lr_by_idxs(optimizer, opt_cfg.lr_decay_ratio, opt_cfg.min_lr, [tcode_pg_idx] )

            #---------------------------------------
            if scfg.check_grad and it%100==0 : 


                for jj in range(len(pm_groups)):
                    
                    gl = grad_log[jj]

                    pg_name, pg = pm_groups[jj]
                    
                    for ii,ppx in enumerate(pg):
                        pp_name, pp =ppx
                        if pp.grad is not None  and (pp.grad!=0).any() :
                            abs_grad=pp.grad.abs() 
                            nz_grad =abs_grad[abs_grad!=0] 

                            assert not torch.isnan(abs_grad).any()
                            # w_name = f'{pg_name}/{pp_name}_mean'

                            self.tbwt.add_scalar(
                                mode='train', 
                                group_name=f'{tag}grad_log',
                                name=f'{tag}grad_log_{pg_name}/{pp_name}',
                                value=nz_grad.mean().item(), 
                                step=accum_step) 

                        else: 
                            pass

            if opt_cfg.pose_vis_interval!=-1 and accum_step % opt_cfg.pose_vis_interval ==0 : 

                sname = f'pose/'
                pose_sub_save_dir = os.path.join( self.vis_sdir, sname)
                os.makedirs(pose_sub_save_dir,exist_ok=True) 

                self.opt_vis_pose(sub_save_dir=pose_sub_save_dir,  
                                  prefix=tag,
                                  cur_fidx=None,
                                  obj_kf_poses=obj_kf_poses,
                                  obj_uids=obj_uids,
                                  obj_idx2tidx=obj_idx2t,
                                  step=accum_step,
                                  post_fix='' ) 

            if opt_cfg.snapshot_interval != -1 and it > 0 and it % opt_cfg.snapshot_interval==0 :
                
                save_fp = os.path.join(self.ckptsdir,
                                        f'{tag}_{accum_step:05d}.pch')

                self.save(save_fp,
                            self.network,
                            obj_kf_poses ,  
                            sce_models,
                            step=accum_step )

            if opt_cfg.vis_interval!=-1 and it>0 and \
                    it% opt_cfg.vis_interval==0 : 

                sname = f'{tag}_{it:05d}'
                sub_save_dir = os.path.join( self.vis_sdir, sname)
                os.makedirs(sub_save_dir,exist_ok=True)

                if scfg.vis_all and enable_vis_all: 
                    self.opt_vis_all( sub_save_dir=sub_save_dir, 
                                      prefix=tag,
                                      keyframes=vis_keyframes,  
                                      keyframes_indices=vis_keyf_indices,
                                      obj_sce_data_list=sce_models, 
                                      obj_bounds_world=obj_bounds, 
                                      obj_kf_poses=obj_kf_poses,
                                      obj_idx2tidx=obj_idx2t,
                                      no_mp4=1, 
                                      step=accum_step )
                
                if scfg.vis_obj and enable_vis_obj: 
                    for k in range(len(sce_models)): 
                        k_obj_uid      = obj_uids[k]
                        k_obj_model    = sce_models[k] 
                        k_obj_bounds   = obj_bounds[k].unsqueeze(0)
                        k_obj_pose     = obj_kf_poses[k] 
                        k_obj_idx2t    = obj_idx2t[k] 

                        self.opt_vis_obj( sub_save_dir=sub_save_dir,
                                          prefix=tag,
                                          opt_cfg=opt_cfg,
                                          keyframes=vis_keyframes,  
                                          keyframes_indices=vis_keyf_indices,
                                          sce_data=k_obj_model, 
                                          bounds_world=k_obj_bounds, 
                                          kf_poses=k_obj_pose,
                                          idx2tidx=k_obj_idx2t,
                                          obj_uid=k_obj_uid,
                                          no_mp4=1, 
                                          step=accum_step,
                                        )   
            #------------------------------------
            timer.tend()

            if early_stop_th != -1 and total_loss.item() < early_stop_th:
                print('early_stop')
                break 
                
        #---------------------------------------------------- 

        self.handle_loss_log(loss_log=loss_log, name_prefix=tag )

        save_fp = os.path.join(self.ckptsdir,
                            f'{tag}_{accum_step:05d}.pch')

        self.save(  save_fp, 
                    self.network,
                    obj_kf_poses ,  
                    sce_models,
                    step=accum_step )

        sname = f'{tag}_{accum_step:05d}'
        sub_save_dir = os.path.join( self.vis_sdir, sname)
        os.makedirs(sub_save_dir,exist_ok=True)

        if scfg.vis_all and enable_vis_all: 
            self.opt_vis_all( sub_save_dir=sub_save_dir, 
                              prefix=tag,
                              keyframes=vis_keyframes,  
                              keyframes_indices=vis_keyf_indices,
                              obj_sce_data_list=sce_models, 
                              obj_bounds_world=obj_bounds, 
                              obj_kf_poses=obj_kf_poses,
                              obj_idx2tidx=obj_idx2t,
                              no_mp4=1, 
                              step=accum_step )
        
        if scfg.vis_obj and enable_vis_obj: 

            for k in range(len(sce_models)): 
                k_obj_uid      = obj_uids[k]
                k_obj_model    = sce_models[k] 
                k_obj_bounds   = obj_bounds[k].unsqueeze(0)
                k_obj_pose     = obj_kf_poses[k] 
                k_obj_idx2t    = obj_idx2t[k] 

                self.opt_vis_obj( sub_save_dir=sub_save_dir,
                                  prefix=tag,
                                  opt_cfg=opt_cfg,
                                  keyframes=vis_keyframes,  
                                  keyframes_indices=vis_keyf_indices,
                                  sce_data=k_obj_model, 
                                  bounds_world=k_obj_bounds, 
                                  kf_poses=k_obj_pose,
                                  idx2tidx=k_obj_idx2t,
                                  obj_uid=k_obj_uid,
                                  no_mp4=1, 
                                  step=accum_step )  

    def opt_mapping(self, sce_models, obj_kf_poses, opt_cfg, keyframes, keyframes_indices  ):

        scfg = self.shared_cfg

        load_ck = self.reload_name is not None 

        if load_ck :
            ck_fp  = os.path.join(self.ckptsdir, self.reload_name+'.pch' )
            print('load checkpoint:',ck_fp)
            start_iters = self.load(ck_fp,
                                    self.network,
                                    obj_kf_poses,
                                    sce_models)

        elif opt_cfg.pre_train_wt is not None: 
            print('load pre-train wt:\n',opt_cfg.pre_train_wt)
            load_fp = opt_cfg.pre_train_wt
            start_iters = self.load(  load_fp,
                                    self.network,
                                    obj_kf_poses,
                                    sce_models)
        else:
            start_iters = 0

        #----------------------------------------
        device = self.device
        sce_models = self.tx_sce_data(sce_models, device)
        total_fnum = len(keyframes)

        vis_keyframes=[]
        vis_keyf_indices=[]

        for i in scfg.vis_frames_indices:
            vis_keyframes.append(keyframes[i])
            vis_keyf_indices.append(keyframes_indices[i])

        #----------------------------------------
        # run scene opt  

        if opt_cfg.run_joint:
            jt_opt_cfg = SimpleNamespace(**opt_cfg.joint)

            obj_bounds = scfg.g_obj_bounds.clone()

            rt = self.f2f_map(start_iters=start_iters,
                              total_iters=jt_opt_cfg.total_iters,
                              sce_models=sce_models,
                              obj_kf_poses=obj_kf_poses,
                              obj_bounds=obj_bounds,
                              obj_uids=scfg.g_obj_uids,
                              obj_idx2t=scfg.g_obj_idx2t,
                              opt_cfg=jt_opt_cfg, 
                              new_opt_keyframes=keyframes, 
                              new_opt_keyf_indices=keyframes_indices,
                              old_opt_keyframes=[],
                              old_opt_keyf_indices=[],
                              vis_keyframes=vis_keyframes,
                              vis_keyf_indices=vis_keyf_indices,
                              do_BA=False,
                              tag=f'map_')


    #==========================================================
    def run_db_init(self, mode):
        device = self.device

        scfg = self.shared_cfg

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        xx = (xx)/(self.W-1)*2.0-1
        yy = (yy)/(self.H-1)*2.0-1
        
        uv_coord = np.stack([xx,yy], axis=2)
        # h w c
        uv_coord = torch.from_numpy(uv_coord).to(torch.float32)
        #uv_coord = rearrange(uv_coord,'h w c -> c h w')

        if mode =='train':
            frame_reader = self.frame_reader_train
            #tkft = self.tracker_ft_train # b h w c
        elif mode =='valid':
            frame_reader = self.frame_reader_valid
            #tkft = self.tracker_ft_val
        else:
            raise Exception('error mode:'+mode)


        frames={}  
        for idx in self.opt_frames_indices:

            # h,w,3
            input_color,input_depth=frame_reader.load_imgs(idx)

            if scfg.segment_mode=='siammask':
                input_seg  = frame_reader.load_siammask_seg(idx)
            elif scfg.segment_mode=='GT':
                input_seg  = frame_reader.load_seg(idx)
            else:
                raise Exception('error')

            objs_pose  = frame_reader.load_my_poses(idx)

            input_nv  = frame_reader.load_normal(idx)

            #--------------
            objs_mask = {} 
            for uid in scfg.g_obj_uids:

                mask = input_seg == uid
                mask = mask.cpu().numpy().astype(np.float32)

                mask2 = dilate(mask,1,10)
                mask2 = torch.from_numpy(mask2>0).to(input_seg.device)
                objs_mask[uid]=mask2

            #-------------- 
            objs_ex_mask = {} 
            for uid in scfg.g_obj_uids:

                mask = input_seg == uid
                mask = mask.cpu().numpy().astype(np.float32)

                mask2 = dilate(mask,4,10)
                mask2 = torch.from_numpy(mask2>0).to(input_seg.device)
                objs_ex_mask[uid]=mask2
            

            #--------------
            input_depth = depth_filter(input_depth)

            if 0:
                # b,c,h,w
                i_tkft = tkft[idx].unsqueeze(0)
                img_size=(H,W)

                tk_ft2 = torch.nn.functional.interpolate(i_tkft, img_size,
                            mode='bicubic')

                tk_ft2 = rearrange(tk_ft2,'1 c h w -> h w c ')

            # input_depth = input_depth.clamp(self.near_z, self.far_z)
            # too_far   = torch.abs(input_depth)>self.far_z
            # too_close = torch.abs(input_depth)<self.near_z 

            # input_depth[too_close]=0
            # input_depth[too_far]  =0

            #'fwflow':fwflow,
            #'bkflow':bkflow,
            # est_ini_poses_v1
            #--------------
            _poses_v1 = frame_reader.load_est_poses_v1(idx)

            ky={'idx':idx,
                'color':input_color,
                'depth':input_depth,
                'seg'  :input_seg,
                'objs_mask':objs_mask,
                'objs_ex_mask':objs_ex_mask, 
                'objs_pose': objs_pose,  # dict, take UID
                'est_ini_poses_v1': _poses_v1,
                'normal':input_nv,
            }

            frames[idx] = ky

        return frames

    def run_model_init(self):

        device = self.device
        scfg = self.shared_cfg

        #-----------------------------------------------
        # build model
        sce_models=[] 

        #assert len(scfg.g_obj_voxel_sizes) == scfg.g_obj_num

        for k in range(scfg.g_obj_num):

            minb = scfg.g_obj_bounds[k,0]
            maxb = scfg.g_obj_bounds[k,1]

            vs = scfg.ray_step_size[k]
            
            minb = torch.tensor(minb,dtype=torch.float32).reshape(1,3)
            maxb = torch.tensor(maxb,dtype=torch.float32).reshape(1,3)

            uid = scfg.g_obj_uids[k]

            # owa = {'geo_radius_init':new_r}
            owa = None

            s=self.sce_factory( min_bound=minb, 
                                max_bound=maxb, 
                                ray_step_size=vs, 
                                overwrite_args=owa )
            sce_models.append(s)


        #-----------------------------------------------
        # K,F

        o_fidxs = self.opt_frames_indices

        obj_kf_poses = []

        for k in range(scfg.g_obj_num):

            uid = scfg.g_obj_uids[k]

            k_fnum = len(o_fidxs)

            if  scfg.g_obj_is_rigid[k] : 
                spec = self.rigid_posenet_spec 
            else: 
                spec = self.nonrigid_posenet_spec
            
            #-------
            _mod   = __import__(spec['module'], fromlist=[spec['class']])
            _arch  = getattr(_mod, spec['class'])
            _poses = _arch(frame_num=k_fnum,**spec['args']) 
            _poses = _poses.to(device) 

            obj_kf_poses.append(_poses)

        #-----------------------------------------------

        total_pnum = 0

        for kpose in obj_kf_poses: 
            pnum = count_parameters(kpose)
            print(f'[{"pose":10}]\tsize={pnum/1000:.1f} K')
            total_pnum+=pnum

        for s in sce_models: 
            pnum1 = count_parameters(s['model'])
            print(f'[{"sce":10}]\tsize={pnum1/1000:.1f} K')
            total_pnum+=pnum1


        #-----------------------------------------------
        pnum = count_parameters(self.network)
        total_pnum+=pnum

        print(f'[{"decoder":10}]\tsize={pnum/1000:.1f} K')
        print('')
        #-----------------------------------------------

        rt={ 
            'sce_models':sce_models,
            'obj_kf_poses':obj_kf_poses,
        } 

        return rt 

    def run_build_poses(self, frame_reader, pose_init_mode, gt_pose_mode):

        scfg = self.shared_cfg

        # scfg.gt_pose_mode
        # pinit_m = scfg.pose_init_mode
        gp_m = gt_pose_mode
        pinit_m = pose_init_mode

        # numpy
        obj_gt_poses  = defaultdict(list)
        obj_ini_poses = defaultdict(list)

        k_fidxs = self.opt_frames_indices

        for k in range(len(scfg.g_obj_uids)):

            #k_pose  = obj_kf_poses[k]
            k_uid   = scfg.g_obj_uids[k] 

            k_start_idx  = k_fidxs[0] 
            #gt_start_c2w = frames[k_start_idx]['objs_pose'][k_uid]
            gt_start_c2w     = frame_reader.load_my_poses(k_start_idx)[k_uid]
            gt_start_c2w_inv = torch.inverse(gt_start_c2w)

            for i,idx in enumerate(k_fidxs):

                #gt_c2w = frames[idx]['objs_pose'][k_uid]
                world_gt_c2w  = frame_reader.load_my_poses(idx)[k_uid]
                world_gt_c2w_np = world_gt_c2w.clone().detach().cpu().numpy()

                nv_gt_c2w = gt_start_c2w_inv @ world_gt_c2w_np
                nv_gt_c2w_np = nv_gt_c2w.clone().detach().cpu().numpy()

                #------------------------------------------------
                if gp_m == 'frame0':
                    obj_gt_poses[k_uid].append(nv_gt_c2w_np)
                elif gp_m == 'world':
                    obj_gt_poses[k_uid].append(world_gt_c2w_np)
                else:
                    raise Exception('error') 
                #------------------------------------------------
                if pinit_m=='GT':  
                    _c2w = obj_gt_poses[k_uid][-1].copy()
                    obj_ini_poses[k_uid].append(_c2w)

                elif pinit_m=='1stframe':
                    #c2w_ns = frames[k_start_idx]['objs_pose'][k_uid] 
                    c2w  = frame_reader.load_my_poses(k_start_idx)[k_uid]
                    obj_ini_poses[k_uid].append(c2w.cpu().numpy())

                elif pinit_m=='eye': 
                    obj_ini_poses[k_uid].append(np.eye(4))

                elif pinit_m=='est_pose_v1':  
                    #c2w = frames[idx]['est_ini_poses_v1'][k_uid]
                    c2w = frame_reader.load_est_poses_v1(idx)[k_uid]
                    obj_ini_poses[k_uid].append(c2w.cpu().numpy())

                else:
                    raise Exception('error [pose_init_mode]='+pinit_m)

        return obj_gt_poses, obj_ini_poses


    def run_train(self): 

        scfg = self.shared_cfg
        device = self.device

        #----------------------------------------
        frames = self.run_db_init(mode='train')

        frames_indices=[]
        for idx in frames.keys(): 
            frames_indices.append(idx)   

        opt_frames=[] 
        opt_frames_indices=frames_indices.copy()

        for idx in opt_frames_indices:
            opt_frames.append(frames[idx]) 
        
        #----------------------------------------
        ini = self.run_model_init()

        sce_models   = ini['sce_models']
        obj_kf_poses = ini['obj_kf_poses']

        #----------------------------------------
        # set rigid pose
        obj_gt_poses, obj_ini_poses = self.run_build_poses(
                                        self.frame_reader_train,
                                        scfg.pose_init_mode,
                                        scfg.gt_pose_mode)

        self.obj_gt_poses  = obj_gt_poses
        self.obj_ini_poses = obj_ini_poses
        
        # set pose block  
        pinit_m = scfg.pose_init_mode

        for k in range(len(scfg.g_obj_uids)): 
            k_pose  = obj_kf_poses[k]
            k_uid   = scfg.g_obj_uids[k] 
            
            for i,idx in enumerate(self.opt_frames_indices): 
                _c2w =obj_ini_poses[k_uid][i]
                _c2w =torch.from_numpy(_c2w).to(torch.float32).to(device)

                k_pose.ini_rigid_pose(i, _c2w)  

        #----------------------------------------
        sce_models = self.tx_sce_data(sce_models, self.device)

        if 1: 
            self.tbwt=writer.TBOverlayWriters(exp_name='', 
                                              log_dir=self.tb_sdir, 
                                              only_train=1)

            if self.args.stage is None :
                opt_mode_list = self.opt_mode
            else:
                opt_mode_list = [self.args.stage]
                #assert self.args.stage in self.opt_cfgs_dict

            print('[opt_mode]:')
            for om in opt_mode_list:
                print(om)
            print('')
            print('======================================')

            opt_steps = len(opt_mode_list)

            for ii in range(opt_steps):

                om = opt_mode_list[ii]

                print('======================================')
                opt_cfg = self.opt_cfgs_dict[om]

                if om=='mapping':
                    rt =self.opt_mapping( 
                                      sce_models,
                                      obj_kf_poses,
                                      opt_cfg, 
                                      opt_frames, 
                                      opt_frames_indices )
                

                elif om=='frame2frame':
                    rt =self.opt_frame2frame( sce_models,
                                              obj_kf_poses,
                                              opt_cfg, 
                                              opt_frames, 
                                              opt_frames_indices,   
                                            )
                else:
                    raise Exception('error:'+om)

        #----------------------------------------
        return


    def run_eval(self, run_train=True, run_val=True, dw=1): 
        eval_fidx = [ x for x in range(0,len(self.opt_frames_indices), self.opt_eval_fidx_step) ]
        
        if run_train:
            self.eval_helper('train',eval_fidx, dw)
        
        if run_val:
            self.eval_helper('valid',eval_fidx, dw)  


    def get_train2val_tx(self):
        scfg = self.shared_cfg
        device = self.device 

        start_idx=self.opt_frames_indices[0]

        gt_start_c2w  = self.frame_reader_train.load_my_poses(start_idx)[0]
        gt_start_c2w_inv = torch.inverse(gt_start_c2w).numpy()
        inv_tr0_tx = gt_start_c2w_inv

        val_gt_poses, _ = self.run_build_poses(
                                    self.frame_reader_valid,
                                    'GT',
                                    'world')

        # i_pose= inv_tr0_tx @  val_obj_gt_world_poses[u][i] 

        new_poses={}
        for u in val_gt_poses:
            
            poses =val_gt_poses[u]

            new_poses[u]=[]

            for i in range(len(poses)):
                tx=poses[i]

                new_tw = inv_tr0_tx @ tx 
                new_poses[u].append(new_tw)

        return new_poses

    def eval_helper(self, data_mode, eval_fidxs, dw): 
        scfg = self.shared_cfg
        device = self.device 

        #----------------------------------------
        ini = self.run_model_init()

        sce_models   = ini['sce_models']
        obj_kf_poses = ini['obj_kf_poses']

        sce_models = self.tx_sce_data(sce_models, self.device)

        #---------------------------------------- 
        load_ck = self.reload_name is not None 
        assert load_ck

        if load_ck :
            ck_fp  = os.path.join(self.ckptsdir, self.reload_name+'.pch' )
            print('load checkpoint:',ck_fp)

            self.load(  ck_fp,
                        self.network,
                        obj_kf_poses,
                        sce_models)

        #----------------------------------------
        use_m2f_training = self.opt_cfgs_dict['mapping'].joint['use_middle_to_fine_training']

        if use_m2f_training: 
            self.network.set_stage('color')
        
        #----------------------------------------
        # reset poses 
        if data_mode=='train':
            frame_reader = self.frame_reader_train
            pass 
        
        elif data_mode=='valid':

            frame_reader = self.frame_reader_valid
            
            assert scfg.gt_pose_mode == 'frame0'
            new_val_poses = self.get_train2val_tx()

            #obj_gt_poses, _ = self.run_build_poses(
            #                                frame_reader,
            #                                'GT',
            #                                'frame0')
            
            # reset all poses 
            for k in range(len(scfg.g_obj_uids)): 
                k_pose = obj_kf_poses[k]
                k_uid  = scfg.g_obj_uids[k] 
                
                for i,idx in enumerate(self.opt_frames_indices): 
                    #_c2w =obj_ini_poses[k_uid][i]
                    _c2w =new_val_poses[k_uid][i]
                    _c2w =torch.from_numpy(_c2w).to(torch.float32).to(device)

                    k_pose.ini_rigid_pose(i, _c2w)  
        else:
            raise Exception('error')  


        #----------------------------------------
        # validation 

        def render_all(idx,slam,obj_kf_poses,sce_models):  
            
            scfg = slam.shared_cfg
            obj_bounds  =scfg.g_obj_bounds
            obj_idx2tidx=scfg.g_obj_idx2t
            
            rt= slam.renderer.render_img(
                            obj_sce_data_list=sce_models,
                            obj_bounds_world=obj_bounds, 
                            obj_idx2tidx=obj_idx2tidx,
                            f_idx=idx,
                            decoder=slam.network,  
                            obj_kf_poses=obj_kf_poses,
                            device=slam.device,
                            near_z=scfg.near_z,
                            far_z =scfg.far_z,
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None,
                            new_dw_step=dw )
            
            return rt 

        def render_obj(idx, slam, obj_kf_poses,sce_models):  
            
            scfg = slam.shared_cfg
            obj_bounds  =scfg.g_obj_bounds
            obj_idx2tidx=scfg.g_obj_idx2t
            
            obj_rts=[]
            for obj_idx in range(len(scfg.g_obj_uids)):
                rt= slam.renderer.render_img(
                            obj_sce_data_list=[sce_models[obj_idx]],
                            obj_bounds_world=obj_bounds[obj_idx].unsqueeze(0), 
                            obj_idx2tidx=[obj_idx2tidx[obj_idx]],
                            f_idx=idx,
                            decoder=slam.network,  
                            obj_kf_poses=[obj_kf_poses[obj_idx]],
                            device=slam.device,
                            near_z=scfg.near_z,
                            far_z =scfg.far_z,
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None,
                            new_dw_step=dw )
                
                obj_rts.append(rt)
            
            return obj_rts 
        
        #----------------------------------------

        render_sdir = f'{self.output}/eval/{self.reload_name}'
        os.makedirs(render_sdir,exist_ok=True)

        pkl_save_fp = os.path.join(render_sdir, f'{data_mode}_v4.pkl')
        print('start')

        obj_render_res=[]
        comb_render_res=[] 
        idx_list=[]

        for idx in tqdm(eval_fidxs):

            idx_list.append(idx)

            rt1 = render_all(idx, self, obj_kf_poses, sce_models)
            comb_render_res.append(rt1)
            

            if len(sce_models)>1:
                rt2 = render_obj(idx, self, obj_kf_poses, sce_models)   
                obj_render_res.append(rt2) 

        #------------------------------------
        with open(pkl_save_fp,'wb') as fout:

            render_res={
                'idx_list':idx_list,
                'comb_render_res':comb_render_res,
            }

            if len(sce_models)>1:
                render_res['obj_render_res']=obj_render_res
        
            pickle.dump(render_res, fout)


        print(pkl_save_fp)
        print('===================================')
        print('all done')

    #-----------------------------------------------------------
    def run_render(self, mode, fnum=None, steps=1, downsample=1):
        assert mode in ['train','valid' ]

        fidx = self.opt_frames_indices
        if steps>1:
            fidx = fidx[::steps]
        if fnum!=None:
            fidx = fidx[:fnum]

        print('render frames:\n', fidx)

        
        if mode=='train':
            self.render_helper('train',fidx, dw=downsample) 
        elif mode=='valid':
            self.render_helper('valid',fidx, dw=downsample)  
        else:
            raise Exception('')
    #-----------------------------------------------------------
    def render_helper(self, data_mode, eval_fidxs, dw):
        scfg = self.shared_cfg
        device = self.device 

        #----------------------------------------
        ini = self.run_model_init()

        sce_models   = ini['sce_models']
        obj_kf_poses = ini['obj_kf_poses']

        sce_models = self.tx_sce_data(sce_models, self.device)

        #---------------------------------------- 
        load_ck = self.reload_name is not None 
        assert load_ck

        ck_fp  = os.path.join(self.ckptsdir, self.reload_name+'.pch' )
        print('load checkpoint:',ck_fp)

        self.load(  ck_fp,
                    self.network,
                    obj_kf_poses,
                    sce_models)

        #----------------------------------------
        # reset poses 
        if data_mode=='train' or data_mode =='tr_debug' or data_mode =='train_q':
            frame_reader = self.frame_reader_train
        
        elif data_mode=='valid' or data_mode =='valid_q':

            frame_reader = self.frame_reader_valid
            
            assert scfg.gt_pose_mode == 'frame0'
            new_val_poses = self.get_train2val_tx()
            
            # reset all poses 
            for k in range(len(scfg.g_obj_uids)): 
                k_pose = obj_kf_poses[k]
                k_uid  = scfg.g_obj_uids[k] 
                
                for i,idx in enumerate(self.opt_frames_indices): 
                    #_c2w =obj_ini_poses[k_uid][i]
                    _c2w =new_val_poses[k_uid][i]
                    _c2w =torch.from_numpy(_c2w).to(torch.float32).to(device)

                    k_pose.ini_rigid_pose(i, _c2w)  
        else:
            raise Exception('error')  
       

        #----------------------------------------
        # render utils 

        def render_all(idx,slam,obj_kf_poses,sce_models):  
            
            scfg = slam.shared_cfg
            obj_bounds  =scfg.g_obj_bounds
            obj_idx2tidx=scfg.g_obj_idx2t
            
            rt= slam.renderer.render_img(
                            obj_sce_data_list=sce_models,
                            obj_bounds_world=obj_bounds, 
                            obj_idx2tidx=obj_idx2tidx,
                            f_idx=idx,
                            decoder=slam.network,  
                            obj_kf_poses=obj_kf_poses,
                            device=slam.device,
                            near_z=scfg.near_z,
                            far_z =scfg.far_z,
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None,
                            new_dw_step=dw )
            
            return rt 

        def render_obj(idx, slam, obj_kf_poses,sce_models):  
            
            scfg = slam.shared_cfg
            obj_bounds  =scfg.g_obj_bounds
            obj_idx2tidx=scfg.g_obj_idx2t
            
            obj_rts=[]
            for obj_idx in range(len(scfg.g_obj_uids)):
                rt= slam.renderer.render_img(
                            obj_sce_data_list=[sce_models[obj_idx]],
                            obj_bounds_world=obj_bounds[obj_idx].unsqueeze(0), 
                            obj_idx2tidx=[obj_idx2tidx[obj_idx]],
                            f_idx=idx,
                            decoder=slam.network,  
                            obj_kf_poses=[obj_kf_poses[obj_idx]],
                            device=slam.device,
                            near_z=scfg.near_z,
                            far_z =scfg.far_z,
                            render_sample_mode=scfg.ray_sample_mode,
                            sample_num=scfg.ray_sample_num,
                            render_shading=self.render_shading,
                            gt_depth=None,
                            new_dw_step=dw  )


                obj_rts.append(rt)
            
            return obj_rts 
        
        #----------------------------------------

        render_sdir = f'{self.output}/render_v4/{data_mode}/{self.reload_name}'
        os.makedirs(render_sdir,exist_ok=True)

        pkl_save_fp = os.path.join(render_sdir, f'{data_mode}_v4.pkl')
        print('start')
        print('save at:',render_sdir)

        obj_render_res=[]
        comb_render_res=[] 
        idx_list=[]

        for idx in tqdm(eval_fidxs):

            idx_list.append(idx)

            rt1 = render_all(idx, self, obj_kf_poses, sce_models)
            comb_render_res.append(rt1) 

            if len(sce_models)>1:
                rt2 = render_obj(idx, self, obj_kf_poses, sce_models)   
                obj_render_res.append(rt2) 

        #------------------------------------
        with open(pkl_save_fp,'wb') as fout:

            render_res={
                'idx_list':idx_list,
                'comb_render_res':comb_render_res,
            }

            if len(sce_models)>1:
                render_res['obj_render_res']=obj_render_res
        
            pickle.dump(render_res, fout)

        print(pkl_save_fp)
        
        #------------------------------------
        # make gif 
        def make_gif(imlist, save_fp, fps=6):
            images = []
            for fp in imlist:
                images.append(imageio.imread(fp))

            imageio.mimsave( save_fp, images, format='GIF', fps=fps)

        def proc(render_res, obj_idx, rgb_save_fp,z_save_fp):

            cmap=plt.get_cmap('plasma')
            #----------------
            _rgb_list=[]
            _z_list=[]
            for i in range(len(render_res)):

                if obj_idx is None:
                    rt = render_res[i]
                else:
                    rt = render_res[i][obj_idx]

                p_color=rt['color'].numpy().copy()
                p_depth=rt['depth'].numpy().copy()

                p_color=p_color*255
                vrgb=p_color.astype(np.uint8)

                vz = cmap(p_depth/10)[:,:,:3] 
                vz = vz*255
                vz = vz.astype(np.uint8)

                _rgb_list.append(vrgb)
                _z_list.append(vz)

            #---------------- 
            with open(rgb_save_fp,'w') as fout:
                pass 
            with open(z_save_fp,'w') as fout:
                pass  

            imageio.mimsave( rgb_save_fp, 
                             _rgb_list, 
                             format='GIF', fps=6)

            imageio.mimsave( z_save_fp, 
                             _z_list,
                             format='GIF', fps=6)
            print(rgb_save_fp) 

        #----------------
        all_rgb_sfp = os.path.join(render_sdir, 
                                       f'gfull_{data_mode}_v4_rgb.gif')
        all_z_sfp   = os.path.join(render_sdir, 
                                       f'gfull_{data_mode}_v4_z.gif')

        proc(comb_render_res, obj_idx=None,
             rgb_save_fp=all_rgb_sfp, z_save_fp=all_z_sfp)
        
        #----------------
        obj_n = len(scfg.g_obj_uids)
        for k in range(obj_n):
            k_rgb_sfp = os.path.join(render_sdir, 
                                           f'gobj{k}_{data_mode}_v4_rgb.gif')
            k_z_sfp   = os.path.join(render_sdir, 
                                           f'gobj{k}_{data_mode}_v4_z.gif')
            proc(obj_render_res, 
                 obj_idx=k, rgb_save_fp=k_rgb_sfp, z_save_fp=k_z_sfp) 

        print('===================================')
        print('all done')
        print('===================================')
        

