
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from   mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import time
import pdb 
import argparse 
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange, reduce, repeat
import open3d as o3d 

from . import o3dtool
import imageio

colors=[(0, 0, 0),
        (34,177,76),
        (0,162,232),
        (255,201,14),
        (201,133,201),
        (64,128,128),  ]

colors=torch.tensor(colors)*1.0/255

cmap_name='jet'


import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.editor import VideoClip


def make_video( imlist, save_fp, fps): 
    
    clip = ImageSequenceClip(imlist, fps=fps)
    
    #clip = VideoClip( imlist, duration=fps)
    
    clip.write_videofile(save_fp, verbose=False, logger=None)
    

def make_gif(imlist, save_fp, fps):

    images = []
    for fp in imlist:
        images.append(imageio.imread(fp))

    imageio.mimsave( save_fp, images, format='GIF', fps=fps)

def smooth(scalars, weight):
    #https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot

    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_grad_log(grad_log, save_fp): 


    keys=list(grad_log.keys())
    keys.sort()

    cands=[]
    for tk in keys:
        vals=grad_log[tk]

        if len(vals)>0 and 'mean' in tk:
            avg= vals 
            tk2= tk.replace('_mean','_std')
            std=grad_log[tk2]
            cands.append((tk,avg,std))

    num = len(cands)
    if num==0:
        # print('[empty log]:', save_fp)
        return
    #-----------------------
    fig, axs =  plt.subplots(1, num, figsize=(6*num,6), dpi=200)
    fig.tight_layout()

    for i in range(num):
        tk,avg,std = cands[i]
            

        if len(avg)>10:
            avg=avg[::10]
            std=std[::10]

        xx = np.arange(len(avg))
        yy = np.asarray(avg) 
        std = np.asarray(std)

        axs[i].plot(xx,yy, linewidth=2)
        axs[i].plot(xx,smooth(yy,0.97), '--', linewidth=2)
        # axs[i].fill_between(xx, yy-std, yy+std, alpha=0.5)

        axs[i].set_title(tk)
        axs[i].set_yscale('log')  
    
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight') 
    
    plt.close('all')
    del fig 


def vis_and_save_render(gt_color_np, gt_depth_np, pred_color_np, pred_depth_np, save_fp):


    depth_residual = np.abs(gt_depth_np - pred_depth_np)
    val_mask = gt_depth_np != 0.0 
    depth_residual[~val_mask]=0.0

    val_dep_rd = depth_residual[val_mask] 
    depth_err  = val_dep_rd.mean()

    color_residual = np.abs(gt_color_np - pred_color_np)
    color_residual = color_residual.mean(axis=-1)  
    color_err = color_residual.mean()  

    #=====================================================================
    fig, axs = plt.subplots(2, 3, figsize=(18,12), dpi=600)
    fig.tight_layout()
    max_depth = np.max(gt_depth_np)

    axs[0, 0].imshow(gt_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    axs[0, 0].set_title('Input Depth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(pred_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    axs[0, 1].set_title('Generated Depth')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    

    if 1:
        _im=axs[0, 2].imshow(depth_residual, cmap="viridis", vmin=0, vmax=0.04)

        divider = make_axes_locatable(axs[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(_im, cax=cax, orientation='vertical')

        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xlabel(f'e={depth_err:.2e}')

    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    

    axs[1, 0].imshow(gt_color_np)
    axs[1, 0].set_title('Input RGB')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(pred_color_np)
    axs[1, 1].set_title('Generated RGB')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    
    if 1:       
        _im=axs[1, 2].imshow(color_residual, cmap="plasma", vmin=0, vmax=0.2)
        _ax=axs[1, 2]
        divider = make_axes_locatable(_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(_im, cax=cax, orientation='vertical')

        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xlabel(f'e={color_err:.2e}')

    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    plt.subplots_adjust(wspace=0.01, hspace=0)
    
    plt.savefig(save_fp, bbox_inches='tight', dpi=300) 
    # plt.clf()
    plt.close('all')


def save_depth_vis(gt_depth_np, save_fp):

    fig, axs = plt.subplots(1, 1, figsize=(5,5), dpi=200)
    fig.tight_layout()
    max_depth = np.max(gt_depth_np)

    axs.imshow(gt_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    axs.set_title('Input Depth')
    axs.set_xticks([])
    axs.set_yticks([])

    plt.savefig(save_fp, bbox_inches='tight', dpi=200) 
    plt.close('all')


def save_obj_pts( obj_pts_list, save_dir, prefix='', postfix='', verbose=False):
        
        if prefix!='':
            prefix+='_'
        if postfix!='':
            postfix=''+postfix
            
        for k in range(len(obj_pts_list)):

            fp = f'{save_dir}/{prefix}obj_k{k:03d}{postfix}.ply'

            pts_np  =obj_pts_list[k].cpu().detach().numpy()

            k_pcd = o3dtool.makePcd(pts_np,  color=colors[k+1])

            o3d.io.write_point_cloud(fp, k_pcd)

            if verbose:
                print(k_pcd)


def vis_alignment( frame_dfts, frame_rgbs, z_codes, 
        frame_xyz, frame_valid_mask, obj_kf_poses, 
        H, W, save_dir, device, prefix='', postfix='' ): 

        #-------------------------------------------------------------

        # b,c,h,w
        #if frame_dfts.shape[2]!=H :
        #    with torch.no_grad():
        #        frame_dfts = F.interpolate(frame_dfts, size=(H,W),mode='bicubic')

        frame_num =frame_dfts.shape[0]

        #-------------------------------------------------------------
        # create segmentation 
        obj_num = z_codes.shape[0]
        
        _,_,SH,SW = frame_dfts.shape
        
        with torch.no_grad():

            # low resolution, 
            # b c sh sw
            frame_dfts_flat = rearrange(frame_dfts,'b c sh sw -> b (sh sw) 1 c')

            z_codes_flat = rearrange(z_codes,'k c -> 1 1 k c').clone().detach() 

            # n,m,k
            likelihood = reduce(frame_dfts_flat*z_codes_flat, 'b x k c-> b x k', 'sum')
            likelihood = torch.exp(likelihood)

            # b x k
            obj_prob = F.softmax(likelihood,dim=-1)
            
            # b,x
            _, obj_seg_vidx = torch.max(obj_prob, dim=-1) 
            obj_seg_vidx = obj_seg_vidx.cpu()

            obj_seg_vidx = rearrange(obj_seg_vidx, 'b (sh sw) -> b sh sw', sh=SH, sw=SW)

            _temp = rearrange(obj_seg_vidx, 'b h w -> b 1 h w').to(torch.float32)

            _temp_up = F.interpolate(_temp, size=(H,W),mode='nearest')


            obj_seg_vidx = rearrange(_temp_up, 'b 1 h w -> b h w')
            obj_seg_vidx = obj_seg_vidx.to(torch.long)

        #-------------------------------------------------------------------------------

        for i in range(frame_num):
            # b c h w
            xyz = frame_xyz[i].cpu()
            rgb = frame_rgbs[i].cpu()
            vm  = frame_valid_mask[i].cpu()

            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd_color = o3d.geometry.PointCloud()

            for k in range(obj_num):

                #pose_i = obj_kf_poses[k][i]()
                #RT = pose_i['RT'].cpu()

                mask = obj_seg_vidx[i] == k    

                obj_mask = vm*mask

                if obj_mask.sum()>0:
                    # obj_p = rearrange(xyz[:,obj_mask],' c n -> 1 n c')
                    obj_p = rearrange(xyz[:,obj_mask],' c n -> n c')
                    obj_p = obj_p.to(device)

                    q = obj_kf_poses[k][i].apply(obj_p, apply_trsl=True)
                    #q = q.squeeze(0).cpu().detach().numpy()
                    q = q.cpu().detach().numpy()
                    q_rgb = rearrange(rgb[:,obj_mask],' c n -> n c').cpu().detach().numpy()

                    k_pcd = o3dtool.makePcd(q,  color=colors[k+1])
                    k_pcd_color = o3dtool.makePcd(q,  color=q_rgb)

                    frame_pcd += k_pcd
                    frame_pcd_color += k_pcd_color
            #-------------------------------------------
            ffp = f'{save_dir}/{prefix}pcd_{i:03d}{postfix}.ply'
            # print(ffp)
            o3d.io.write_point_cloud(ffp,frame_pcd)

            ffp2 = f'{save_dir}/{prefix}colorpcd_{i:03d}{postfix}.ply'
            o3d.io.write_point_cloud(ffp2,frame_pcd_color)
            print(ffp2)


def vis_alignment_tidx( frame_dfts, frame_rgbs, z_codes, 
        frame_xyz, frame_valid_mask, obj_kf_poses, 
        H, W, save_dir, device, prefix='', postfix='' ): 

        #-------------------------------------------------------------

        # b,c,h,w
        #if frame_dfts.shape[2]!=H :
        #    with torch.no_grad():
        #        frame_dfts = F.interpolate(frame_dfts, size=(H,W),mode='bicubic')

        frame_num =frame_dfts.shape[0]

        #-------------------------------------------------------------
        # create segmentation 
        obj_num = z_codes.shape[0]
        
        _,_,SH,SW = frame_dfts.shape
        
        with torch.no_grad():

            # low resolution, 
            # b c sh sw
            frame_dfts_flat = rearrange(frame_dfts,'b c sh sw -> b (sh sw) 1 c')

            z_codes_flat = rearrange(z_codes,'k c -> 1 1 k c').clone().detach() 

            # n,m,k
            likelihood = reduce(frame_dfts_flat*z_codes_flat, 'b x k c-> b x k', 'sum')
            likelihood = torch.exp(likelihood)

            # b x k
            obj_prob = F.softmax(likelihood,dim=-1)
            
            # b,x
            _, obj_seg_vidx = torch.max(obj_prob, dim=-1) 
            obj_seg_vidx = obj_seg_vidx.cpu()

            obj_seg_vidx = rearrange(obj_seg_vidx, 'b (sh sw) -> b sh sw', sh=SH, sw=SW)

            _temp = rearrange(obj_seg_vidx, 'b h w -> b 1 h w').to(torch.float32)

            _temp_up = F.interpolate(_temp, size=(H,W),mode='nearest')


            obj_seg_vidx = rearrange(_temp_up, 'b 1 h w -> b h w')
            obj_seg_vidx = obj_seg_vidx.to(torch.long)

        #-------------------------------------------------------------------------------

        for i in range(frame_num):
            # b c h w
            xyz = frame_xyz[i].cpu()
            rgb = frame_rgbs[i].cpu()
            vm  = frame_valid_mask[i].cpu()

            frame_pcd = o3d.geometry.PointCloud()
            frame_pcd_color = o3d.geometry.PointCloud()

            for k in range(obj_num): 
                mask = obj_seg_vidx[i] == k    

                obj_mask = vm*mask

                k_obj_pose = obj_kf_poses[k]
                #tx_p = k_obj_pose.apply( t_idx=i, p=src_xyz,  apply_trsl=True) 
                #tx_p = k_obj_pose.apply_inv( t_idx=j, p=tx_p, apply_trsl=True)

                if obj_mask.sum()>0:
                    #obj_p = rearrange(xyz[:,obj_mask],' c n -> 1 n c')
                    obj_p = rearrange(xyz[:,obj_mask],' c n -> n c')
                    obj_p = obj_p.to(device)

                    q = k_obj_pose.apply(t_idx=i, p=obj_p, apply_trsl=True)
                    #q = q.squeeze(0).cpu().detach().numpy()
                    q = q.cpu().detach().numpy()
                    q_rgb = rearrange(rgb[:,obj_mask],' c n -> n c').cpu().detach().numpy()

                    k_pcd = o3dtool.makePcd(q,  color=colors[k+1])
                    k_pcd_color = o3dtool.makePcd(q,  color=q_rgb)

                    frame_pcd += k_pcd
                    frame_pcd_color += k_pcd_color
            #-------------------------------------------
            ffp = f'{save_dir}/{prefix}pcd_{i:03d}{postfix}.ply'
            # print(ffp)
            o3d.io.write_point_cloud(ffp,frame_pcd)

            ffp2 = f'{save_dir}/{prefix}colorpcd_{i:03d}{postfix}.ply'
            o3d.io.write_point_cloud(ffp2,frame_pcd_color)
            print(ffp2)


def vis_seg( frame_dfts, frame_rgbs, z_codes, H, W, save_dir, prefix='', postfix='' ): 

        #-------------------------------------------------------------

        # b,c,h,w
        #if frame_dfts.shape[2]!=H :
        #    with torch.no_grad():
        #        frame_dfts = F.interpolate(frame_dfts, size=(H,W),mode='bicubic')

        frame_num =frame_dfts.shape[0]

        #-------------------------------------------------------------
        # create segmentation 
        obj_num = z_codes.shape[0]
        
        _,_,SH,SW = frame_dfts.shape
        
        with torch.no_grad():

            # low resolution, 
            # b c sh sw
            frame_dfts_flat = rearrange(frame_dfts,'b c sh sw -> b (sh sw) 1 c')

            z_codes_flat = rearrange(z_codes,'k c -> 1 1 k c').clone().detach() 

            # n,m,k
            likelihood = reduce(frame_dfts_flat*z_codes_flat, 'b x k c-> b x k', 'sum')
            likelihood = torch.exp(likelihood)

            # b x k
            obj_prob = F.softmax(likelihood,dim=-1)
            
            # b,x
            _, obj_seg_vidx = torch.max(obj_prob, dim=-1)
            obj_seg_vidx+=1
            obj_seg_vidx = obj_seg_vidx.cpu()

            obj_seg_vidx = rearrange(obj_seg_vidx, 'b (sh sw) -> b sh sw', sh=SH, sw=SW)

        #-------------------------------------------------------------
        
        obj_seg_vis = torch.zeros(frame_num,SH,SW,3, dtype=torch.float32, device='cpu')

        for b in range(frame_num):
            for k in range(1,obj_num+1):
                mask = obj_seg_vidx[b]== k
                obj_seg_vis[b,mask,:]= colors[k,:]
        
        with torch.no_grad():
            _temp = rearrange(obj_seg_vis, 'b h w c -> b c h w')
            _temp_up = F.interpolate(_temp, size=(H,W),mode='bilinear')
            obj_seg_vis = rearrange(_temp_up, 'b c h w -> b h w c')
            
        #-------------------------------------------------------------
        obj_seg_vis_np = obj_seg_vis.detach().cpu().numpy()

        input_color_np = rearrange(frame_rgbs,'b c h w -> b h w c').clone().detach().cpu().numpy() 

        #-------------------------------------------------------------
        #gt_distll_np = gt_distll.detach().cpu().numpy() 
        # gt_xx = gt_distll_np.reshape(-1,C)
        if 0:
            pca = PCA(n_components=3)
            ft_r = pca.fit(gt_xx).transform(gt_xx)
            min_v=ft_r.min(0)
            max_v=ft_r.max(0)
            gt_distft_vis = (ft_r-min_v)/(max_v-min_v) 
            gt_distft_vis = gt_distft_vis.reshape(IMH,IMW,3)

        #-------------------------------------------------------------

        if len(prefix)>0:
            prefix=prefix+'_'

        if len(postfix)>0:
            postfix='_'+postfix
            
        for i in range(frame_num):

            fig, axs = plt.subplots(1, 2, figsize=(6,12), dpi=200)
            fig.tight_layout() 

            #---------------
            axs[0].imshow( input_color_np[i] )
            axs[0].set_title('Input Color')
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            axs[1].imshow( obj_seg_vis_np[i] )
            axs[1].set_title('Generated Seg.')
            axs[1].set_xticks([])
            axs[1].set_yticks([])
                    
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
            ffp = f'{save_dir}/{prefix}seg_{i:03d}{postfix}.png'

            plt.savefig(ffp, bbox_inches='tight', dpi=300) 
            plt.clf()

            print(f'Saved: {ffp}')

def vis_and_save_render_mask(gt_color_np, gt_depth_np, pred_color_np, pred_depth_np, save_fp, mask):


    depth_residual = np.abs(gt_depth_np - pred_depth_np)
    val_mask = (gt_depth_np != 0.0) * mask

    depth_residual[~val_mask]=0.0 
    val_dep_rd = depth_residual[val_mask]  
    depth_err  = val_dep_rd.mean()

    color_residual = np.abs(gt_color_np - pred_color_np)
    color_residual = color_residual.mean(axis=-1)  
    val_mask = mask

    color_residual[~val_mask]=0.0 
    val_rgb_rd = color_residual[val_mask]  
    color_err  = val_rgb_rd.mean() 

    #=====================================================================
    fig, axs = plt.subplots(2, 3, figsize=(18,12), dpi=600)
    fig.tight_layout()
    max_depth = np.max(gt_depth_np)

    axs[0, 0].imshow(gt_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    axs[0, 0].set_title('Input Depth')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(pred_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    axs[0, 1].set_title('Generated Depth')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    

    if 1:
        _im=axs[0, 2].imshow(depth_residual, cmap="viridis", vmin=0, vmax=0.04)

        divider = make_axes_locatable(axs[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(_im, cax=cax, orientation='vertical')

        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xlabel(f'e={depth_err:.2e}')

    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    

    axs[1, 0].imshow(gt_color_np)
    axs[1, 0].set_title('Input RGB')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    
    axs[1, 1].imshow(pred_color_np)
    axs[1, 1].set_title('Generated RGB')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    
    if 1:       
        _im=axs[1, 2].imshow(color_residual, cmap="plasma", vmin=0, vmax=0.2)
        _ax=axs[1, 2]
        divider = make_axes_locatable(_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(_im, cax=cax, orientation='vertical')

        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xlabel(f'e={color_err:.2e}')

    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    plt.subplots_adjust(wspace=0.01, hspace=0)
    
    plt.savefig(save_fp, bbox_inches='tight', dpi=300) 
    # plt.clf()
    plt.close('all')




def vis_and_save_alpha(gt_color_np, pred_aa, save_fp):

    K = pred_aa.shape[-1]

    #=====================================================================
    fig, axs = plt.subplots(1, K+1, figsize=(6*K,12), dpi=600)
    fig.tight_layout()

    axs[0].imshow(gt_color_np)
    axs[0].set_title('Input Color')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    for i in range(K):
        axs[i+1].imshow(pred_aa[:,:,i], cmap=cmap_name, vmin=0, vmax=1.0)
        axs[i+1].set_title(f'Alpha channel-{i}')
        axs[i+1].set_xticks([])
        axs[i+1].set_yticks([])
    
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(save_fp, bbox_inches='tight', dpi=300) 
    # plt.clf()
    plt.close('all')



def vis_geo_and_save_render_mask(gt_color_np, gt_depth_np, pred_color_np, pred_depth_np, pred_shading_np, save_fp, mask):


    depth_residual = np.abs(gt_depth_np - pred_depth_np)
    val_mask = (gt_depth_np != 0.0) 

    if mask is not None:
        val_mask = val_mask*mask

    depth_residual[~val_mask]=0.0 
    val_dep_rd = depth_residual[val_mask]  
    depth_err  = val_dep_rd.mean()

    color_residual = np.abs(gt_color_np - pred_color_np)
    color_residual = color_residual.mean(axis=-1) 

    if mask is not None:
        val_mask = mask
        color_residual[~val_mask]=0.0  
        val_rgb_rd = color_residual[val_mask]  
    else:
        val_rgb_rd = color_residual
        
    color_err  = val_rgb_rd.mean() 

    #=====================================================================
    fig, axs = plt.subplots(1, 3, figsize=(18,12), dpi=600)
    fig.tight_layout()
    max_depth = np.max(gt_depth_np)

    # axs[0, 0].imshow(gt_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    # axs[0, 0].set_title('Input Depth')
    # axs[0, 0].set_xticks([])
    # axs[0, 0].set_yticks([])

    i_ax = axs[0]
    i_ax.imshow(pred_color_np)
    i_ax.set_title('Pred. RGB')
    i_ax.set_xticks([])
    i_ax.set_yticks([])
    i_ax.set_xlabel(f'err:{color_err:.2e}')
    
    if pred_shading_np is not None:
        i_ax = axs[1]
        i_ax.imshow(pred_shading_np, cmap=cmap_name, vmin=0, vmax=max_depth)
        i_ax.set_title('Pred. Shading')
        i_ax.set_xticks([])
        i_ax.set_yticks([])

    i_ax = axs[2]
    i_ax.imshow(pred_depth_np, cmap=cmap_name, vmin=0, vmax=max_depth)
    i_ax.set_title('Pred. Depth')
    i_ax.set_xticks([])
    i_ax.set_yticks([])
    i_ax.set_xlabel(f'err:{depth_err:.2e}')
    
    if 0:
        _im=axs[0, 2].imshow(depth_residual, cmap="viridis", vmin=0, vmax=0.04)

        divider = make_axes_locatable(axs[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(_im, cax=cax, orientation='vertical')

        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xlabel(f'e={depth_err:.2e}')


    #=================
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(save_fp, bbox_inches='tight', dpi=200) 
    plt.close('all')

