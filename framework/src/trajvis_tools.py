import numpy as np
import os,sys,time

import torch
import torch.nn.functional as torch_F

import torchvision
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from easydict import EasyDict as edict
import pdb 


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)
def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)
def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)
def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)

    if lim is not None:
        ax.set_xlim(lim.x[0],lim.x[1])
        ax.set_ylim(lim.y[0],lim.y[1])
        ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev, azim=azim) 

def get_camera_mesh(pose,scale):

    vertices = np.array([[-0.5,-0.5,1],
                         [0.5,-0.5,1],
                         [0.5,0.5,1],
                         [-0.5,0.5,1],
                         [0,0,0]])*scale

    vertices = vertices.reshape(1,-1,3) 
    vertices = np.repeat(vertices, pose.shape[0], axis=0)
    vertices = vertices.transpose(0,2,1) 

    # ijk * ikl = ijl
    vertices = np.matmul( pose[:,:3,:3], vertices )

    vertices = vertices + pose[:,:3,3:]
    vertices = vertices.transpose(0,2,1)

    faces =  np.array([[0,1,2],
                      [0,2,3],
                      [0,1,4],
                      [1,2,4],
                      [2,3,4],
                      [3,0,4]])

    center = vertices[:,-1]

    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return center,vertices, faces, wireframe

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged

def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged

def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged


def plot_draw_poses(pose, pose_ref=None, cam_scale=0.04,
                    save_fp=None, return_fig=False, lim1=None,lim2=None ):
    
    # get the camera meshes
    cam_c,_,_,cam = get_camera_mesh(pose,scale=cam_scale) 

    if pose_ref is not None:
        cem_c_ref,_,_,cam_ref = get_camera_mesh(pose_ref,
                            scale=cam_scale)
        #cam_ref = cam_ref.numpy()
    
    # set up plot window(s)
    fig=plt.figure(figsize=(12,6),dpi=100)

    # plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121,projection="3d")
    ax2 = fig.add_subplot(122,projection="3d")
    
    setup_3D_plot(ax1, elev=-90,azim=-90, lim=lim1 )
    setup_3D_plot(ax2, elev=45 ,azim=-45, lim=lim2 )

    # ax1.set_title("forward-facing view",pad=0)
    # ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)

    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)

            ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=4)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=4)

        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=4)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=4)

    ax1.plot(cam_c[:,0],cam_c[:,1],cam_c[:,2],color='r')
    ax2.plot(cam_c[:,0],cam_c[:,1],cam_c[:,2],color='r')
    
    ax1.plot(cem_c_ref[:,0],cem_c_ref[:,1],cem_c_ref[:,2],color='b')
    ax2.plot(cem_c_ref[:,0],cem_c_ref[:,1],cem_c_ref[:,2],color='b')
    #ax1.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')

    _ax=ax1
    lim1=edict(x=_ax.get_xlim(),y=_ax.get_ylim(),z=_ax.get_zlim())
    _ax=ax2
    lim2=edict(x=_ax.get_xlim(),y=_ax.get_ylim(),z=_ax.get_zlim())

    if save_fp is not None:
        plt.savefig(save_fp,dpi=150)
        
    if return_fig:
        return lim1,lim2,fig 
    else:
        plt.close('all')
        del fig 
        return lim1,lim2


def plot_draw_poses_v2(pose, pose_ref=None, cam_scale=0.04,
                    save_fp=None, return_fig=False, 
                    lim1=None,lim2=None,
                    dotsize=4 ):
    
    # get the camera meshes
    cam_c,_,_,cam = get_camera_mesh(pose,scale=cam_scale) 

    if pose_ref is not None:
        cem_c_ref,_,_,cam_ref = get_camera_mesh(pose_ref,
                            scale=cam_scale)
        #cam_ref = cam_ref.numpy()
    
    # set up plot window(s)
    #fig=plt.figure(figsize=(6,6),dpi=100)

    # plt.title("epoch {}".format(ep))
    #ax1 = fig.add_subplot(111,projection="3d")
    
    fig=plt.figure(figsize=(6,6),dpi=100)
    ax2 = fig.add_subplot(111,projection="3d")
    
    #setup_3D_plot(ax1, elev=-90,azim=-90, lim=lim1 )
    setup_3D_plot(ax2, elev=45 ,azim=-45, lim=lim2 )

    # ax1.set_title("forward-facing view",pad=0)
    # ax2.set_title("top-down view",pad=0)
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)

    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):

        if pose_ref is not None:
            #ax1.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=(0.3,0.3,0.3),linewidth=1)
            ax2.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],
                    color=(0.3,0.3,0.3),linewidth=1)

            #ax1.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=dotsize)
            ax2.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=(0.3,0.3,0.3),s=dotsize)

        #c = np.array(color(float(i)/N))*0.8
        c = np.array([1.0,0.0,0.0])
        #ax1.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c)
        #ax1.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=dotsize)
        ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=dotsize)

    #ax1.plot(cam_c[:,0],cam_c[:,1],cam_c[:,2],color='r')
    ax2.plot(cam_c[:,0],cam_c[:,1],cam_c[:,2],color='r')
    
    #ax1.plot(cem_c_ref[:,0],cem_c_ref[:,1],cem_c_ref[:,2],color='b')
    ax2.plot(cem_c_ref[:,0],cem_c_ref[:,1],cem_c_ref[:,2],color='b')
    #ax1.set_aspect('equal', 'box')
    #ax2.set_aspect('equal', 'box')

    #_ax=ax1
    #lim1=edict(x=_ax.get_xlim(),y=_ax.get_ylim(),z=_ax.get_zlim())
    _ax=ax2
    lim2=edict(x=_ax.get_xlim(),y=_ax.get_ylim(),z=_ax.get_zlim())

    if save_fp is not None:
        plt.savefig(save_fp,dpi=150, bbox_inches="tight")
        
    if return_fig:
        return lim2,fig 
    else:
        plt.close('all')
        del fig 
        return lim2



def plot_draw_poses_v3_single(fig, ax2, pose_est_list, pcolor, cam_scale=0.04,
                    save_fp=None, return_fig=False, 
                    lim1=None,lim2=None,
                    dotsize=4 ):
    
    # get the camera meshes
    for pose in pose_est_list:
        cam_c,_,_,cam = get_camera_mesh(pose,scale=cam_scale)  

        N = len(cam)
        color = plt.get_cmap("gist_rainbow")
        for i in range(N): 
            c = pcolor
            ax2.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=c,linewidth=0.5)
            ax2.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=c,s=dotsize)

        ax2.plot(cam_c[:,0],cam_c[:,1],cam_c[:,2],color='r',linewidth=0.5)
        
    #_ax=ax2
    #lim2=edict(x=_ax.get_xlim(),y=_ax.get_ylim(),z=_ax.get_zlim())

    if save_fp is not None:
        plt.savefig(save_fp,dpi=150, bbox_inches="tight")
        
    # if return_fig:
    #     return lim2,fig 
    # else:
    #     plt.close('all')
    #     del fig 
    #     return lim2

def plot_pose_errbar(pose, pose_ini, pose_gt, save_fp=None, return_fig=False):

    # list 
    fig = plt.figure(dpi=100)

    num = len(pose)

    #----------------------------------------
    est_rot_errs =[]
    est_trsl_errs=[]

    for i in range(num): 
        r_err = np.abs(pose[i][:3,:3]-pose_gt[i][:3,:3]).mean()
        est_rot_errs.append(r_err)

        t_err = np.abs(pose[i][:3,3]-pose_gt[i][:3,3]).mean()
        est_trsl_errs.append(t_err)

    #----------------------------------------
    ini_rot_errs =[]
    ini_trsl_errs=[]

    for i in range(num): 
        r_err = np.abs(pose_ini[i][:3,:3]-pose_gt[i][:3,:3]).mean()
        ini_rot_errs.append(r_err)

        t_err = np.abs(pose_ini[i][:3,3]-pose_gt[i][:3,3]).mean()
        ini_trsl_errs.append(t_err)

    #---------------------------------------
    xx= np.arange(num)

    fig=plt.figure(figsize=(24,12),dpi=100)
    plt.subplot(121)
    plt.bar(xx-0.1,ini_rot_errs, width=0.2,color='b',      label='Initial')
    plt.bar(xx+0.1,est_rot_errs, width=0.2,color='orange', label='Optimzed')
    
    plt.legend()
    plt.title('Rotation')
    plt.xlabel('Frame ID')

    plt.xticks(xx)
    plt.yscale('log')

    plt.subplot(121)
    plt.bar(xx-0.1,ini_trsl_errs, width=0.2,color='b',      label='Initial')
    plt.bar(xx+0.1,est_trsl_errs, width=0.2,color='orange', label='Optimzed')
    plt.legend()

    plt.title('Translation')
    plt.xlabel('Frame ID')

    plt.xticks(xx)
    plt.yscale('log')

    #-----------------------------------------
    if save_fp is not None:
        plt.savefig(save_fp,dpi=150)
        
    if return_fig:
        return fig 
    else:
        plt.close('all')
        del fig  

def plot_pose_errbar_v2(pose, pose_ini, pose_gt, save_fp=None, return_fig=False, cut=False):
    
    fig = plt.figure(dpi=100) 
    num = len(pose)

    #----------------------------------------
    est_rot_errs =[]
    est_trsl_errs=[]

    for i in range(num): 
        r_err = np.abs(pose[i][:3,:3]-pose_gt[i][:3,:3]).mean()
        est_rot_errs.append(r_err)

        t_err = np.abs(pose[i][:3,3]-pose_gt[i][:3,3]).mean()
        est_trsl_errs.append(t_err)

    #----------------------------------------
    ini_rot_errs =[]
    ini_trsl_errs=[]

    for i in range(num): 
        r_err = np.abs(pose_ini[i][:3,:3]-pose_gt[i][:3,:3]).mean()
        ini_rot_errs.append(r_err)

        t_err = np.abs(pose_ini[i][:3,3]-pose_gt[i][:3,3]).mean()
        ini_trsl_errs.append(t_err)

    #---------------------------------------

    xx= np.arange(num)
    scale=2.5
    wd   =1.2
    shift=wd+wd*0.2

    if cut:
        xx=xx[::2]
        ini_rot_errs=ini_rot_errs[::2]
        est_rot_errs=est_rot_errs[::2]

        ini_trsl_errs=ini_trsl_errs[::2]
        est_trsl_errs=est_trsl_errs[::2]

    #---------------------------------------

    scale_xt = (xx*scale)[::10].tolist()
    scale_xtlabel=(xx)[::10].tolist()

    scale_xtlabel=[str(v) for v in scale_xtlabel]

    fig=plt.figure(figsize=(15,5), dpi=100)
    plt.subplot(211)
    plt.bar(xx*scale,  ini_rot_errs,   
            width=wd,  color='b',      label='Initial' )
    plt.bar(xx*scale+shift,est_rot_errs, 
            width=wd,  color='orange', label='Optimzed')
    
    plt.legend()
    plt.title( 'Rotation')
    plt.xlabel('Frame ID')


    plt.xticks(scale_xt, labels=scale_xtlabel)
    plt.yscale('log')
    #---------------------------------------
    plt.subplot(212)
    plt.bar(xx*scale, ini_trsl_errs,       
            width=wd, color='b',      label='Initial')
    plt.bar(xx*scale+shift,est_trsl_errs, 
            width=wd, color='orange', label='Optimzed')
    plt.legend()

    plt.title('Translation')
    plt.xlabel('Frame ID')

    plt.xticks(scale_xt, labels=scale_xtlabel)

    plt.yscale('log')
    plt.subplots_adjust(hspace=0.5)

    #-----------------------------------------
    if save_fp is not None:
        plt.savefig(save_fp,dpi=150)
        
    if return_fig:
        return fig 
    else:
        plt.close('all')
        del fig  

