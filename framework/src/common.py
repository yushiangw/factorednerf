import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import pdb 
import os 

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])



def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack([ (i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs*c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape) 
    return rays_o, rays_d


def get_rays_from_uv_cam_space(i, j, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    #if isinstance(c2w, np.ndarray):
    #    c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    #dirs = dirs.reshape(-1, 1, 3)
    rays_d = dirs.reshape(-1, 3)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    #rays_d = torch.sum(dirs*c2w[:3, :3], -1)
    #rays_o = c2w[:3, -1].expand(rays_d.shape)
    rays_o = torch.zeros_like(dirs)
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)

    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    
    depth = depth[indices]  # (n)
    color = color[indices]  # (n,3)
    
    return i, j, depth, color


def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, sample_depth, sample_color = get_sample_uv( H0, H1, W0, W1, n, depth, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color


def get_samples_v2(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """

    iw, ih, sample_depth, sample_color = get_sample_uv( H0, H1, W0, W1, n, depth, color, device=device)


    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color, iw, ih


def get_samples_v2_my( subsample, H, W, fx, fy, cx, cy, c2w, depth, color, device):
    """
        randomly select samples  and avoid invalid samples 
    """
    #==============================    

    #depth = depth[H0:H1, W0:W1]
    #color = color[H0:H1, W0:W1]

    valid = depth>0
    valid_idx = torch.nonzero(valid)
    valid_ih = valid_idx[:,0]
    valid_iw = valid_idx[:,1]

    VNUM   = valid_idx.shape[0] 

    # n, 
    ii = (torch.rand(subsample, device=device) * VNUM ).long() 
    ih = valid_ih[ii]
    iw = valid_iw[ii]

    sample_d= depth[ih,iw] 
    
    #if (sample_d!=0).sum()>(subsample*0.1):
    #    break

    #-----------------------------------------------------

    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]

    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, fx, fy, cx, cy, device)
    
    return rays_o, rays_d, sample_depth, sample_color, iw, ih


def get_samples_v3_my( subsample, H, W, fx, fy, cx, cy, c2w, depth, color, seg, device):
    """
        sample valid points using depth

    """
    #==============================    

    #depth = depth[H0:H1, W0:W1]
    #color = color[H0:H1, W0:W1]

    valid = depth>0
    valid_idx = torch.nonzero(valid)
    valid_ih = valid_idx[:,0]
    valid_iw = valid_idx[:,1]

    VNUM   = valid_idx.shape[0] 

    # n, 
    ii = (torch.rand(subsample, device=device) * VNUM ).long() 
    ih = valid_ih[ii]
    iw = valid_iw[ii]

    #-----------------------------------------------------

    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]
    sample_seg   = seg[ih,iw]
    
    sample_seg = sample_seg.to(device)
    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, fx, fy, cx, cy, device)
    
    return rays_o, rays_d, sample_depth, sample_color, sample_seg, iw, ih



def get_samples_v4_my( subsample1, subsample2, H, W, fx, fy, cx, cy, c2w, depth, color, p_mask1, p_mask2, seg, device):
    """
        sample point use masj

    """
    #==============================    

    #depth = depth[H0:H1, W0:W1]
    #color = color[H0:H1, W0:W1]

    valid1 = (p_mask1>0) #* (depth>0)
    valid_idx1 = torch.nonzero(valid1)
    valid_ih1 = valid_idx1[:,0]
    valid_iw1 = valid_idx1[:,1]

    VNUM1   = valid_idx1.shape[0] 

    # n, 
    ii1 = (torch.rand( subsample1, device=device) * VNUM1 ).long() 
    ih1 = valid_ih1[ii1]
    iw1 = valid_iw1[ii1]

    #-----------------------------------------------------
    valid2 = (p_mask2>0) #* (depth>0)
    valid_idx2 = torch.nonzero(valid2)
    valid_ih2 = valid_idx2[:,0]
    valid_iw2 = valid_idx2[:,1]

    VNUM2   = valid_idx2.shape[0] 

    # n, 
    ii2 = (torch.rand( subsample2, device=device) * VNUM2 ).long() 
    ih2 = valid_ih2[ii2]
    iw2 = valid_iw2[ii2]

    #-----------------------------------------------------

    ih = torch.cat([ih1,ih2])
    iw = torch.cat([iw1,iw2])

    #-----------------------------------------------------
    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]
    sample_seg   = seg[ih,iw]
    
    sample_seg = sample_seg.to(device)

    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, 
                            fx, fy, cx, cy, device)
    
    return rays_o, rays_d, sample_depth, sample_color, sample_seg, iw, ih


def get_samples_v5_my( subsample1,  H, W, fx, fy, cx, cy, c2w, depth, color, p_mask, seg, device):
    """
        sample point use mask 

    """

    valid1 = (p_mask>0)
    valid_idx1 = torch.nonzero(valid1)
    valid_ih1  = valid_idx1[:,0]
    valid_iw1  = valid_idx1[:,1]

    VNUM1   = valid_idx1.shape[0] 

    # n, 
    ii1 = (torch.rand( subsample1, device=device) * VNUM1 ).long() 
    ih  = valid_ih1[ii1]
    iw  = valid_iw1[ii1]

    #---------------------------------
    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]
    sample_seg   = seg[ih,iw]
    
    sample_seg = sample_seg.to(device)
    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, 
                            fx, fy, cx, cy, device)
    
    return rays_o, rays_d, sample_depth, sample_color, sample_seg, iw, ih


def get_surface_sample_v1( subsample1,  H, W, fx, fy, cx, cy, c2w, depth, color, p_mask, seg, device, tracker_ft=None ):
    """
        sample point use mask 

    """

    valid1 = (p_mask>0)
    valid_idx1 = torch.nonzero(valid1)
    valid_ih1  = valid_idx1[:,0]
    valid_iw1  = valid_idx1[:,1]

    VNUM1   = valid_idx1.shape[0] 

    # n, 
    ii1 = (torch.rand( subsample1, device=device) * VNUM1 ).long() 
    ih  = valid_ih1[ii1]
    iw  = valid_iw1[ii1]

    #---------------------------------
    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]
    sample_seg   = seg[ih,iw]
    
    sample_seg = sample_seg.to(device)

    if tracker_ft is not None:
        sample_tkft  = tracker_ft[ih,iw].to(device)
    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, 
                            fx, fy, cx, cy, device)

    sf_pts = rays_o + rays_d*sample_depth[...,None]

    rt={}
    rt['sample_pts']=sf_pts
    rt['sample_vdirs']=rays_d
    rt['sample_color']=sample_color
    rt['sample_depth']=sample_depth
    rt['sample_seg']=sample_seg
    rt['sample_tkft']=sample_tkft
    rt['iw']=iw
    rt['ih']=ih 
    
    return rt 


def get_surface_sample_v2( subsample1,  H, W, fx, fy, cx, cy, c2w, depth, color, p_mask, seg, device, tracker_ft=None):
    """
        modified steps when # < VNUM

    """

    valid1 = (p_mask>0)
    valid_idx1 = torch.nonzero(valid1)
    valid_ih1  = valid_idx1[:,0]
    valid_iw1  = valid_idx1[:,1]

    VNUM1   = valid_idx1.shape[0] 

    if VNUM1<subsample1:

        num2 = subsample1-VNUM1

        ii1 = (torch.rand( num2, device=device) * VNUM1 ).long() 
        ih  = valid_ih1[ii1]
        iw  = valid_iw1[ii1]

        ih = torch.cat([ih,valid_ih1],dim=0)
        iw = torch.cat([iw,valid_iw1],dim=0)
    else: 
        ridx= torch.randperm(VNUM1)[:subsample1]
        ih  = valid_ih1[ridx]
        iw  = valid_iw1[ridx]


    #---------------------------------
    sample_depth = depth[ih,iw]
    sample_color = color[ih,iw]
    sample_seg   = seg[ih,iw]
    sample_seg = sample_seg.to(device)

    if tracker_ft is not None:
        sample_tkft  = tracker_ft[ih,iw].to(device)

    #-----------------------------------------------------
    rays_o, rays_d = get_rays_from_uv(iw, ih, c2w, H, W, 
                            fx, fy, cx, cy, device)

    sf_pts = rays_o + rays_d*sample_depth[...,None]

    rt={}
    rt['sample_pts']=sf_pts
    rt['sample_vdirs']=rays_d
    rt['sample_color']=sample_color
    rt['sample_depth']=sample_depth
    rt['sample_seg']=sample_seg
    rt['iw']=iw
    rt['ih']=ih 
    
    if tracker_ft is not None:
        rt['sample_tkft']=sample_tkft
    
    return rt 

#=================================================
def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    device = RT.device 
    dtype = RT.dtype 

    # gpu_id = -1
    # if type(RT) == torch.Tensor:
    #     if RT.get_device() != -1:
    #         RT = RT.detach().cpu()
    #         #gpu_id = RT.get_device()
    #         gpu_id = RT.get_device()
    #     RT = RT.numpy()
    if type(RT) == torch.Tensor:
        RT = RT.detach().cpu().numpy()

    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)

    tensor = torch.from_numpy(tensor).to(dtype).to(device)

    #if gpu_id != -1:
    #    tensor = tensor.to(gpu_id) 

    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy=False, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    rgb = raw[..., :-1]

    if occupancy:
        raw[..., 3] = torch.sigmoid(10*raw[..., -1])
        alpha = raw[..., -1]
    else:
        def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
            torch.exp(-act_fn(raw)*dists)
            
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = dists.float()
        dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
            device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        # different ray angle corresponds to different unit length
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def raw2outputs_nerf_color_v2(raw, z_vals, rays_d, occupancy=False, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    rgb = raw[..., :-1]

    #if occupancy:
    #    raw[..., 3] = torch.sigmoid(10*raw[..., -1])
    #    alpha = raw[..., -1]
    if occupancy:
        #raw[..., 3] = torch.sigmoid(10*raw[..., -1])
        alpha = raw[..., -1]
    else:
        def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
            torch.exp(-act_fn(raw)*dists)
            
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = dists.float()
        dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
            device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        # different ray angle corresponds to different unit length
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_v2(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)

    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d, i, j 


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p

#=================================================
def ray_sdf_to_alpha(z_quert_st, uniform_ssize, ray_s, ray_d, decoder, sce_model, cos_anneal_ratio): 
    
    assert z_quert_st.ndim==2 
    assert ray_s.ndim==2
    assert ray_d.ndim==2

    N = ray_s.shape[0]

    dists = z_quert_st[:, 1:] - z_quert_st[:, :-1]
    # [t,extra_dist]

    dists = torch.cat([dists, uniform_ssize], -1)
    mid_z_vals = z_quert_st + dists * 0.5

    # Section midpoints
    # n,t,3
    pts  = ray_s[:, None, :] + ray_d[:, None, :] * mid_z_vals[:, :, None]  
    dirs = ray_d[:, None, :].expand(pts.shape) 
    
    pts  = pts.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)
    
    # pred. SDF+COLOR  
    out = decoder(pts, scene_model=sce_model, return_grad=True)
    rgb = out['rgb']
    sdf = out['sdf']
    gradients  = out['nv'] 
    inv_s = out['inv_s']

    # n*t
    true_cos = (dirs * gradients).sum(-1, keepdim=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # Estimate signed distances at section points
    dists = dists.reshape(-1, 1) 

    estimated_next_sdf = sdf + iter_cos * dists* 0.5
    estimated_prev_sdf = sdf - iter_cos * dists* 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5))
    alpha = alpha.clip(0.0, 1.0)

    alpha = rearrange(alpha, '(n m) 1 -> n m', n=N)
    rgb   = rearrange(rgb, '(n m) c -> n m c', n=N)
    
    return alpha, rgb, sdf, mid_z_vals


def ray_sdf_to_alpha_v3( z, z_len, ray_s, ray_d, decoder, sce_model, cos_anneal_ratio): 
    
    assert z.ndim==2  
    assert ray_s.ndim==2
    assert ray_d.ndim==2

    # N,M
    assert z_len.shape == z.shape
    N = z.shape[0]
    #====================================================== 

    # Section midpoints, n,m,3
    pts  = ray_s[:, None, :] + ray_d[:, None, :] * z[:, :, None]  
    dirs = ray_d[:, None, :].expand(pts.shape) 
    
    pts  = pts.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)
    
    # pred. SDF+COLOR  
    out = decoder(pts, scene_model=sce_model, return_grad=True)
    rgb = out['rgb']
    sdf = out['sdf']
    gradients  = out['nv'] 
    inv_s = out['inv_s']

    # n*m,1
    true_cos = (dirs * gradients).sum(-1, keepdim=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # Estimate signed distances at section points
    z_len = rearrange(z_len, 'n m -> (n m) 1')

    estimated_next_sdf = sdf + iter_cos * z_len* 0.5
    estimated_prev_sdf = sdf - iter_cos * z_len* 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5))
    alpha = alpha.clip(0.0, 1.0)

    alpha = rearrange(alpha, '(n m) 1 -> n m', n=N)
    rgb   = rearrange(rgb, '(n m) c -> n m c', n=N)
    
    return alpha, rgb, sdf


def raw2outputs_nerf_SDF_color(rgb, z_vals, alpha, device):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    weights_sum = weights.sum(dim=-1, keepdim=True)

    rgb_map   = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)

    return depth_map, depth_var, rgb_map, weights


def ray_occ_to_alpha( z, ray_s, ray_d, decoder, sce_model): 
    
    assert z.ndim==2  
    assert ray_s.ndim==2
    assert ray_d.ndim==2
    
    N = z.shape[0]
    #====================================================== 

    # Section midpoints, n,m,3
    pts  = ray_s[:, None, :] + ray_d[:, None, :] * z[:, :, None]  
    dirs = ray_d[:, None, :].expand(pts.shape) 
    
    pts  = pts.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)


    out = decoder(pts, scene_model=sce_model)
    rgb = out['rgb']
    occ = out['occ'] 
    
    alpha = occ 

    alpha = rearrange(alpha, '(n m) 1 -> n m'  , n=N )
    rgb   = rearrange(rgb,   '(n m) c -> n m c', n=N )
    
    return alpha, rgb, occ


def raw2outputs_nerf_Occ_color(rgb, z_vals, alpha, device):
    """
    """
    assert alpha.ndim==2
    assert z_vals.ndim==2

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)

    return depth_map, depth_var, rgb_map, weights



#=================================================

def t_quat2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def quat2matrix(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = t_quat2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def matrix2quat_nogd(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    device = RT.device 
    dtype  = RT.dtype 

    if type(RT) == torch.Tensor:
        RT = RT.detach().cpu().numpy()

    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)

    tensor = torch.from_numpy(tensor).to(dtype).to(device)
    
    return tensor


def plot_loss(loss_list, names, save_fp): 
    np_losses = []
    for ll in loss_list:
        np_losses.append(np.asarray(ll))

    num=len(np_losses)

    #-----------------------
    fig, axs =  plt.subplots(1, num, figsize=(4*num,4), dpi=72)
    fig.tight_layout()

    # ytrange=[1.0,1.0e-1,1.0e-2,1.0e-3,1.0e-4,1.0e-5,1.0e-6,0.0]
    # ytnames =[ f'{y:.1e}' for y in ytrange ]

    for i in range(num):
        axs[i].plot(np_losses[i])
        axs[i].set_title(names[i])
        axs[i].set_yscale('log')  
        # axs[i].set_yticks( ytrange)
        # axs[i].set_yticklabels(ytnames)
        # axs[i].set_ylim(0.0,1.0)
        axs[i].grid(True)

        # formatter = ticker.FuncFormatter(lambda y, _: f'{y:.3g}')
        # ax.xaxis.set_major_formatter(formatter) 
    
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight') 
        plt.clf() 
    del fig 


def plot_loss_v2(loss_list, names, save_fp): 
    np_losses = []
    for ll in loss_list:
        np_losses.append(np.asarray(ll))

    num=len(np_losses)

    #-----------------------
    fig, axs =  plt.subplots(1, num, figsize=(4*num,4), dpi=72)
    fig.tight_layout()

    ytrange=[1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0]
    ytnames =[ f'{y:.1e}' for y in ytrange ]

    for i in range(num):
        axs[i].plot(np_losses[i])

        if '[par]' not in names[i]:
            axs[i].set_title(names[i])
            axs[i].set_yscale('log')  
            axs[i].set_yticks( ytrange )
            axs[i].set_yticklabels(ytnames)
            axs[i].set_ylim(1.0e-6, 1.0)
            axs[i].grid(True)

        # formatter = ticker.FuncFormatter(lambda y, _: f'{y:.3g}')
        # ax.xaxis.set_major_formatter(formatter) 
    
    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight') 
        plt.clf() 
    del fig 

def save_ray_log(ray_img, save_fp2):
    assert save_fp2 is not None 


    fig, axs =  plt.subplots(1, 1, figsize=(6,6), dpi=200)
    fig.tight_layout()

    ray_img_np = ray_img.cpu().numpy()

    axs.imshow( ray_img_np )   
    
    plt.savefig(save_fp2, bbox_inches='tight') 
    plt.clf()  
    del fig 


def random_nd_index(num, subsample):
    ridx = (torch.rand(subsample) * (num-1)).long()
    return ridx 

def torch_load(fp, map_location):
    
    assert os.path.exists(fp), fp 

    x = torch.load(fp, map_location=map_location)

    ld = {}
    for tk in x:
        ld[tk]=x[tk]

    return ld 