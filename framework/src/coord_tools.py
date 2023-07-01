import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import pdb


def valid_backproj_v2( c2w, color, depth, H0, H1, W0, W1, fx, fy,  cx,  cy, use_neg_z, flip_y, return_img=False):
    #
    # Tensor
    #
    # return:
    #
    #   p_c: N,3
    #   p_rgb: N,3
    #

    device = depth.device

    iz = depth.clone()
    valid = iz!=0 
    
    ix, iy = torch.meshgrid(torch.linspace(W0, W1-1, W1-W0), torch.linspace(H0, H1-1, H1-H0))
    ix=ix.t() 
    iy=iy.t() 

    ix=ix[valid].to(torch.float32)
    iy=iy[valid].to(torch.float32)
    iz=iz[valid].to(torch.float32)
    
    p_rgb=color[valid]

    if ix.numel()==0:
        print('zero valid pts')
        return torch.tensor(()) 

    xx = (ix-cx)/fx
    yy = (iy-cy)/fy
    zz = torch.ones_like(ix)

    if flip_y:
        yy *= -1

    if use_neg_z:
        zz *= -1

    p_c = torch.stack([xx,yy,zz], dim=-1) 
    p_c = p_c.to(device)
    p_c = p_c.t()
    p_c = p_c*iz[None,:]
    
    p_d = torch.matmul(c2w[:3, :3],p_c) 
    p_c= p_d + c2w[:3, -1,None]
    p_c= p_c.t()
    
    if return_img:
        p_im_rgb = torch.zeros((depth.shape[0],depth.shape[1],3),dtype=torch.float32,device=device)
        p_im_xyz = torch.zeros((depth.shape[0],depth.shape[1],3),dtype=torch.float32,device=device)

        p_im_rgb[valid]=p_rgb
        p_im_xyz[valid]=p_c

        return p_im_xyz, p_im_rgb, valid
    else:
        return p_c, p_rgb, valid

def valid_backproj( c2w, color, depth, H0, H1, W0, W1, fx, fy,  cx,  cy, return_img=False):
    #
    # Tensor
    #
    # return:
    #
    #   p_c: N,3
    #   p_rgb: N,3
    #

    device = depth.device

    iz = depth.clone()
    valid = iz!=0 
    
    ix, iy = torch.meshgrid(torch.linspace(W0, W1-1, W1-W0), torch.linspace(H0, H1-1, H1-H0))
    ix=ix.t() 
    iy=iy.t() 

    ix=ix[valid].to(torch.float32)
    iy=iy[valid].to(torch.float32)
    iz=iz[valid].to(torch.float32)
    
    p_rgb=color[valid]

    if ix.numel()==0:
        return torch.tensor(()) 

    p_c = torch.stack([(ix-cx)/fx, -(iy-cy)/fy, -torch.ones_like(ix)], dim=-1)
    p_c = p_c.to(device)
    p_c = p_c.t()
    p_c = p_c*iz[None,:]
    
    p_d = torch.matmul(c2w[:3, :3],p_c) 
    p_c= p_d + c2w[:3, -1,None]
    p_c= p_c.t()
    
    if return_img:
        p_im_rgb = torch.zeros((depth.shape[0],depth.shape[1],3),dtype=torch.float32,device=device)
        p_im_xyz = torch.zeros((depth.shape[0],depth.shape[1],3),dtype=torch.float32,device=device)

        p_im_rgb[valid]=p_rgb
        p_im_xyz[valid]=p_c

        return p_im_xyz, p_im_rgb, valid
    else:
        return p_c, p_rgb, valid


def valid_backproj_simple( c2w, depth, H0, H1, W0, W1, fx, fy,  cx,  cy, return_img=False):
    #
    # Tensor
    #
    # return:
    #
    #   p_c: N,3
    #   p_rgb: N,3
    

    device = depth.device
    
    ix, iy = torch.meshgrid(
                    torch.linspace(W0, W1-1, W1-W0), 
                    torch.linspace(H0, H1-1, H1-H0))
    # h,w
    ix=ix.t() 
    iy=iy.t() 

    #------------------------------------
    iz = depth.clone()
    valid = iz!=0 
    
    # h,w,c 
    p_c = torch.stack([(ix-cx)/fx, -(iy-cy)/fy, -torch.ones_like(ix)], dim=-1)
    p_c = p_c.to(device) 
    p_c = p_c*iz[...,None]
    
    shape=p_c.shape 
    
    p_c = p_c.reshape(-1,3).T

    p_d = torch.matmul(c2w[:3, :3],p_c) 
    p_c= p_d + c2w[:3,-1,None]
    # h*w,3
    p_c= p_c.t()
    p_c= p_c.reshape(shape)    

    return p_c,valid

def world2image_np(pts, c2w, fx, fy,  cx,  cy):

    assert pts.ndim==2
    # world2camera
    w2c = np.linalg.inv(c2w)
    ones = np.ones_like(pts[:, 0]).reshape(-1, 1)
    
    homo_vertices = np.concatenate([pts, ones], axis=1).reshape(-1, 4, 1)

    cam_cord_homo = w2c@homo_vertices
    # N,3,1
    cam_cord = cam_cord_homo[:, :3]
    
    # invert x 
    cam_cord[:, 0] *= -1
    
    K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)

    uv = K@cam_cord
    # N,1,1
    z  = uv[:, -1:]#+1e-5
    # N,2,1
    uv = uv[:, :2]

    nonzero = (z!=0.0).squeeze()
    uv[nonzero] = uv[nonzero]/z[nonzero]
    uv = uv.astype(np.float32)

    # N,2,1
    return uv, z 


def camera2image(pts, fx, fy,  cx,  cy):

    assert pts.ndim==2

    device=pts.device
    
    # world2camera
    #w2c = np.linalg.inv(c2w)
    #ones = np.ones_like(pts[:, 0]).reshape(-1, 1)    
    #homo_vertices = np.concatenate([pts, ones], axis=1).reshape(-1, 4, 1)
    #cam_cord_homo = w2c@homo_vertices
    # N,3,1
    #cam_cord = cam_cord_homo[:, :3]

    # N,3
    cam_cord = torch.zeros_like(pts)

    # invert x 
    cam_cord[:, 0] = -1*pts[:,0]
    cam_cord[:, 1:]=    pts[:,1:]
    
    # 3,3
    K = torch.tensor( [[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]] ).reshape(3, 3).to(device)

    # N,3 
    uv = K@ (cam_cord.T)
    uv = uv.T

    # N,1
    z  = uv[:,-1] #+1e-5
    zz = z.unsqueeze(-1)

    nonzero = (z!=0.0)

    # N,2
    uv2 = torch.zeros_like(uv[:, :2]) 
    uv2[nonzero] = uv[nonzero,:2]/zz[nonzero] 

    # N,2
    return uv2, nonzero