import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from src.my_model import grad  

from .dyn_sampler import ray_box_intersection
import pdb 
from torch.profiler import profile, record_function, ProfilerActivity

__all__ = [ 'get_sample_mode', 
            'volume_rendering', 
            'rays_search_a_b' ]

SAMPLE_INV_RAND=1
SAMPLE_INV_UNIFORM=2
SAMPLE_INV_UNIRAND=3
SAMPLE_FIX_UNIFORM=4
SAMPLE_UNI_INV_UNIFORM=5
SAMPLE_UNI_INV_UNIRAND=6
SAMPLE_UNI_INV_UNIFORM_x4=7
SAMPLE_INV_UNIFORM_x4=8

def get_sample_mode(mode):
    # if mode=='mix':
    #    return SAMPLE_MIX_INV_UNIFORM

    if mode=='inverse_rand':
        return SAMPLE_INV_RAND

    elif mode=='inverse_uniform':
        return SAMPLE_INV_UNIFORM
        
    elif mode=='inverse_unirand':
        return SAMPLE_INV_UNIRAND
        
    elif mode=='uniform':
        return SAMPLE_FIX_UNIFORM

    elif mode=='uniform_inv_uniform':
        return SAMPLE_UNI_INV_UNIFORM

    elif mode=='uniform_inv_unirand':
        return SAMPLE_UNI_INV_UNIRAND

    elif mode=='uniform_inv_uniform_x4':
        return SAMPLE_UNI_INV_UNIFORM_x4
        
    elif mode=='inv_uniform_x4':
        return SAMPLE_INV_UNIFORM_x4
        
    else:
        raise Exception('error')

def inverse_sample(cdf, z_vals, u):
    assert cdf.ndim==2
    device = cdf.device
    
    assert u.shape[0] == cdf.shape[0]
    
    K=cdf.shape[-1]
    
    inds  = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds),  inds-1)
    above = torch.min((K-1)*torch.ones_like(inds), inds)
    
    inds_g = torch.stack([below, above], dim=-1) 
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    
    cdf_g  = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    z_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, inds_g)
    
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = z_g[..., 0] + t * (z_g[..., 1] - z_g[..., 0])
    
    return samples

def pdf2cdf(pdf,eps=1.0e-6):
    pdf = pdf / (torch.sum(pdf, -1, keepdim=True)+eps)
    cdf = torch.cumsum(pdf, -1)
    return cdf 
    

def neus_sdf2weights(z_vals, sdf, invs, valid):

    device = sdf.device

    N = z_vals.shape[0]
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    _da   = dists[...,-1:].clone().detach()
    dists = torch.cat([dists, _da], dim=-1)
    mid_z_vals = z_vals + dists * 0.5
    
    #-----------------------------------------------
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    _a    = dists[...,-1:].clone().detach()
    # _a = torch.ones_like(dists[:,-1:])*1000

    dists = torch.cat([dists, _a], -1)

    # Estimate signed distances at section points
    #iter_cos =1.0
    #estimated_next_sdf = sdf + iter_cos * dists * 0.5
    #estimated_prev_sdf = sdf - iter_cos * dists * 0.5
    #prev_cdf = torch.sigmoid(estimated_prev_sdf * invs)
    #next_cdf = torch.sigmoid(estimated_next_sdf * invs)

    assert invs.shape == sdf.shape 
    cdf = torch.sigmoid(sdf * invs)

    prev_cdf = cdf[...,:-1]
    next_cdf = cdf[...,1:]

    # *,M-1
    p = prev_cdf - next_cdf
    c = prev_cdf

    # *,M-1
    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

    x = torch.ones((*alpha.shape[:-1], 1)).to(device)

    if valid is not None: 
        # *,M-1
        s_valid = valid[...,:-1]*valid[...,1:]
        assert s_valid.shape == alpha.shape
        alpha2 = torch.zeros_like(alpha)
        alpha2[s_valid] = alpha[s_valid]
        alpha  = alpha2 

    transmit = torch.cumprod( torch.cat([ x, 1.0 - alpha + 1e-7], -1), -1)
    transmit = transmit[...,:-1]

    assert alpha.shape == transmit.shape 
    weights = alpha * transmit
    
    # N,M-1
    return weights, alpha, transmit


def volume_rendering(sdf, rgb, z_vals, invs, valid, device, weight_only=False): 
    # N, M-1 
    weights, alpha, transmit= neus_sdf2weights(z_vals, sdf, invs, valid) 

    if weight_only:
        
        return weights, transmit

    else:
        weights_ = weights.unsqueeze(-1)
        rgb_map   = torch.sum(weights_ * rgb[...,:-1,:],  -2 )  # (N_rays, 3)
        depth_map = torch.sum(weights  * z_vals[...,:-1], -1 )  # (N_rays)
        
        tmp = (z_vals[...,:-1]-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
        depth_var = torch.sum(weights*tmp*tmp, dim=-1)   # (N_rays)

        return depth_map, depth_var, rgb_map, weights,transmit


def z_uniform_samples(N, upsample_num, near, far, device):

    #ray_z_step_len = ((ray_d**2).sum(dim=-1)+1.0e-7).sqrt()

    #1,k
    vals = torch.linspace(0.0, 1.0, steps=upsample_num,device=device)
    vals = vals.unsqueeze(0)

    # n,k
    z_uniform = near[:,None] + vals*((far-near)[:,None])

    return z_uniform


def z_inverse_samples(ray_o, ray_d, ray_ti, ray_near, ray_far, upsample_num, stepsize,   sce_data, decoder, motion_net, deform, use_unifrom_ini,   use_rand_ini, device, max_pre_samples=4096, min_pre_samples=64,iterations=1):
    # 
    # estimate density using stepsize  

    ray_z_step_len = (ray_d**2).sum(dim=-1).sqrt()
    ray_vx_step_size = stepsize/ray_z_step_len
    ray_vx_step_size = rearrange(ray_vx_step_size,'n -> n 1').to(device)

    nn = torch.ceil((ray_far-ray_near)/stepsize).to(torch.int).max()
    
    nn = max(min_pre_samples, min(nn, max_pre_samples))

    # 1,m
    t = torch.linspace(0.0, 1.0, steps=nn,device=device)
    t = t.unsqueeze(0)

    # n,k
    z_uniform = ray_near[:,None] + t*((ray_far-ray_near)[:,None])
    
    #======================================= 
    z_list = [] 
    i_z = z_uniform.sort(dim=-1)[0]

    for _ in range(iterations):

        # F M N
        pts  = ray_o[:, None, :] + ray_d[:, None, :] * i_z[:, :, None] 
        
        N,M = pts.shape[:2]
        
        ptsn3= pts.reshape(-1, 3)  
        
        #dirs = ray_d[:, None, :].expand(pts.shape)
        #dirsn3= dirs.reshape(-1,3)
        #vdirs = torch.nn.functional.normalize(dirsn3,dim=-1)
        vdirs = torch.nn.functional.normalize(ray_d,dim=-1)
        vdirs = repeat(vdirs,'f c -> f n c', n=pts.shape[1])
        vdirs = vdirs.reshape(-1,3)

        # with torch.no_grad(): 
        t_idx = repeat(ray_ti,'n -> (n m)', m=M)

        with torch.no_grad():
            out = decoder(ptsn3,
                          vdirs=vdirs,
                          scene_data=sce_data, 
                          motion_net=motion_net, 
                          p_tidx=t_idx,
                          normalize=True,
                          deform=deform,
                          sdf_only=True)
        
        _sdf = out['sdf']  
        _sdf = rearrange( _sdf, '(n m) 1 -> n m',n=N,m=M)
        
        _invs= out['p_invs']
        _invs = rearrange( _invs, '(n m) 1 -> n m',n=N,m=M)
        #invs = sce_data['model'].get_invs()

        # n,m-1
        weights,_,_ = neus_sdf2weights(i_z, _sdf, _invs, valid=None)

        pdf = weights  
        cdf = pdf2cdf(pdf)
        
        #--------------------------------------- 
        # d_noc_p   =out['_noc_p'].reshape(N,M,3)[100]
        # d_tx_noc_p=out['_tx_noc_p'].reshape(N,M,3)[100]
        # d_valid   =out['_valid'].reshape(N,M)[100]
        # d_wt      =weights[100]
        # d_i_z     =i_z[100]
        #--------------------------------------- 
        M=upsample_num
        
        if use_unifrom_ini and use_rand_ini :
            M1=M//2
            M2=M-M1
            u1 = torch.rand((N,M1),device=device) 
            u2 = torch.linspace(0., 1., steps=M2, device=device) 
            u2 = u2.unsqueeze(0).repeat(N, 1)

            u = torch.cat([u1,u2],dim=-1)  
            u, _ = u.sort(dim=-1)

        elif use_rand_ini :
            u = torch.rand((N,M),device=device)   
            u, _ = u.sort(dim=-1)

        elif use_unifrom_ini:
            u = torch.linspace(0., 1., steps=M, device=device)
            u = u.unsqueeze(0).repeat(N, 1)
        else:
            raise Exception("error")

        #---------------------------------------  
        zz = inverse_sample(cdf, i_z[:,:-1], u)
        # zz = zz[:,:-1]
        
        z_list.append(zz) 
        i_z = zz 

    #=======================================
    # n,k
    z_upsampled = torch.cat(z_list,dim=-1)
    z_upsampled, stidx = z_upsampled.sort(dim=-1)
    z_upsampled = z_upsampled.detach()

    return z_upsampled


def rays_search_z_single_obj(ray_o, ray_d, ray_ti, ray_tnear, ray_tfar, sample_mode, snum, sce_data, decoder, motion_net, 
    deform, near_z, far_z, esti_normal, render_tracker_ft, max_pre_samples,device):

    stepsize = sce_data['ray_step_size']

    # sce_data['min_bound']
    # sce_data['grid_size']

    #==========================================  
    if sample_mode == SAMPLE_FIX_UNIFORM:

        z_upsampled = z_uniform_samples(  ray_o.shape[0], 
                                          snum, 
                                          ray_tnear, ray_tfar, 
                                          device)

    elif sample_mode == SAMPLE_INV_UNIFORM or sample_mode == SAMPLE_INV_UNIRAND:

        use_unifrom_ini = sample_mode == SAMPLE_INV_UNIFORM
        use_rand_ini = sample_mode == SAMPLE_INV_UNIRAND

        z_upsampled = z_inverse_samples(ray_o, ray_d, ray_ti,
                                        ray_near=ray_tnear, 
                                        ray_far=ray_tfar, 
                                        upsample_num=snum, 
                                        stepsize=stepsize, 
                                        sce_data=sce_data, 
                                        decoder=decoder,
                                        motion_net=motion_net,
                                        deform=deform, 
                                        use_unifrom_ini=use_unifrom_ini,
                                        use_rand_ini=use_rand_ini,
                                        max_pre_samples=max_pre_samples,
                                        device=device)
        
    elif sample_mode == SAMPLE_UNI_INV_UNIFORM  or sample_mode == SAMPLE_UNI_INV_UNIRAND:

        z1 = z_uniform_samples( ray_o.shape[0], snum//2, 
                                ray_tnear, ray_tfar, device)

        use_unifrom_ini = True
        use_rand_ini    = sample_mode == SAMPLE_UNI_INV_UNIRAND

        z2 = z_inverse_samples( ray_o, ray_d, ray_ti,
                                ray_near=ray_tnear, 
                                ray_far=ray_tfar, 
                                upsample_num=snum//2, 
                                stepsize=stepsize, 
                                sce_data=sce_data, 
                                decoder=decoder,
                                motion_net=motion_net,
                                deform=deform, 
                                use_unifrom_ini=use_unifrom_ini,
                                use_rand_ini=use_rand_ini,
                                max_pre_samples=max_pre_samples,
                                device=device)

        z_upsampled = torch.cat([z1,z2],dim=-1)
        
        z_upsampled, _ = z_upsampled.sort(dim=-1)

    elif sample_mode == SAMPLE_UNI_INV_UNIFORM_x4:

        uni_num = snum//2
        assert uni_num>4

        z1 = z_uniform_samples( ray_o.shape[0], uni_num, 
                                ray_tnear, ray_tfar, device)

        use_unifrom_ini = True
        use_rand_ini    = False

        # 64/4=16 
        upnum = snum-uni_num 
        upnum = upnum//4
        
        z2 = z_inverse_samples( ray_o, ray_d, ray_ti,
                                ray_near=ray_tnear, 
                                ray_far=ray_tfar, 
                                upsample_num=upnum, 
                                stepsize=stepsize,
                                sce_data=sce_data, 
                                decoder=decoder,
                                motion_net=motion_net,
                                deform=deform, 
                                use_unifrom_ini=use_unifrom_ini,
                                use_rand_ini=use_rand_ini,
                                max_pre_samples=max_pre_samples,
                                device=device,
                                iterations=4)

        z_upsampled = torch.cat([z1,z2],dim=-1)
        z_upsampled, _ = z_upsampled.sort(dim=-1)


    elif sample_mode == SAMPLE_INV_UNIFORM_x4:
        use_unifrom_ini = True
        use_rand_ini    = False

        # 64/4=16 
        upnum = snum//4
        z_upsampled = z_inverse_samples( ray_o, ray_d, ray_ti,
                                ray_near=ray_tnear, 
                                ray_far=ray_tfar, 
                                upsample_num=upnum, 
                                stepsize=stepsize,
                                sce_data=sce_data, 
                                decoder=decoder,
                                motion_net=motion_net,
                                deform=deform, 
                                use_unifrom_ini=use_unifrom_ini,
                                use_rand_ini=use_rand_ini,
                                max_pre_samples=max_pre_samples,
                                device=device,
                                iterations=4)

    else:
        raise Exception('error')

    #=======================================
    N = ray_o.shape[0]

    # f n c
    pts  = ray_o[:, None, :] + ray_d[:, None, :] * z_upsampled[:, :, None] 

    vdirs = torch.nn.functional.normalize(ray_d,dim=-1)
    vdirs = repeat(vdirs,'f c -> f n c', n=pts.shape[1])
    # vdirs = vdirs.expand(pts.shape)

    if esti_normal and not pts.requires_grad:  
        pts.requires_grad=True 


    M=pts.shape[1] 
    t_idx = repeat(ray_ti,'n -> (n m)', m=M)

    pts_n3  = pts.reshape(-1, 3)
    vdirs_n3= vdirs.reshape(-1,3)

    out = decoder(pts_n3,
                  vdirs=vdirs_n3,
                  scene_data=sce_data, 
                  motion_net=motion_net, 
                  p_tidx=t_idx,
                  normalize=True,
                  deform=deform) 

    rgb    = out['rgb']
    sdf    = out['sdf']
    p_invs = out['p_invs']
    valid  = out['_valid']

    sdf= rearrange(sdf,'(n m) 1 -> n m'  , n=N ,m=snum)
    valid= rearrange(valid,'(n m) -> n m'  , n=N ,m=snum)
    rgb  = rearrange(rgb,  '(n m) c -> n m c', n=N ,m=snum)
    p_invs=rearrange(p_invs,'(n m) 1 -> n m'  , n=N ,m=snum)

    if deform:
        p_shift = out['p_shift']
        p_shift = rearrange(p_shift,  '(n m) c -> n m c', n=N ,m=snum)
    else:
        p_shift = None 

    if esti_normal:

        if 'normal' in out and out['normal'] is not None:
            
            #p_grad   = out['normal']
            p_normal = out['normal']

            #p_grad  = rearrange(p_grad,'(n m) c -> n m c'  , n=N ,m=snum)
            p_normal= rearrange(p_normal,'(n m) c -> n m c'  , n=N ,m=snum)
        else:
            p_grad   = grad.gradient(sdf, pts) 
            p_normal = torch.nn.functional.normalize(p_grad,dim=-1)

    else:
        #p_grad = None
        p_normal = None

    if render_tracker_ft:
        p_tkft = out['tracker_ft']
        p_tkft = rearrange(p_tkft,  '(n m) c -> n m c', n=N ,m=snum)
    else:
        p_tkft = None

    out={
        'rgb':rgb,   # N SN c
        'sdf':sdf,   # N SN
        'z':z_upsampled, # N SN 
        'pts':pts, # N SN 3
        'p_invs':p_invs, # N SN
        #'p_grad':p_grad, # N SN 3
        'normal':p_normal,
        'p_shift':p_shift, # N SN 3
        'valid':valid, # N SN 
        'tracker_ft':p_tkft,
    } 
    return out 


def render_rays_w_objs( 
    cam_rays_o, cam_rays_d, f_indices, obj_idx2tidx,
    obj_kf_poses, obj_bounds_world, obj_sce_data_list, 
    decoder,  render_sample_mode, render_sample_num, 
    near_z, far_z, device, 
    esti_normal=False,
    render_obj_vec=False,
    render_tracker_ft=False,
    max_pre_samples=4096,
    verbose=False):
    
    # render_mode,
    assert cam_rays_o.ndim==3
    # f n c  
    rays_o   = cam_rays_o.clone()
    rays_d   = cam_rays_d.clone()

    F,N,C   = rays_o.shape 

    f_num   = F
    obj_num = len(obj_kf_poses)

    assert len(f_indices)==F 

    #============================================================ 
    with record_function("my_rr_apply_rigid"):

        obj_rays_o= torch.zeros((F, N, obj_num, C), device=device )
        obj_rays_d= torch.zeros((F, N, obj_num, C), device=device )
        obj_rays_ti=torch.zeros((F, N, obj_num, ), device=device, dtype=torch.long)

        for k in range(obj_num): 
            k_obj_pose = obj_kf_poses[k]
            k_idx2tidx = obj_idx2tidx[k] 

            for i in range(f_num):
                idx = f_indices[i] 

                if idx in k_idx2tidx:
                    t = k_idx2tidx[idx]
                else:
                    t = -1

                if t != -1:
                    # f n k c
                    obj_rays_o[i,:,k] = k_obj_pose.apply_rigid( t_idx=t, p=rays_o[i], apply_trsl=True) 
                    obj_rays_d[i,:,k] = k_obj_pose.apply_rigid( t_idx=t, p=rays_d[i], apply_trsl=False)  
                
                obj_rays_ti[i,:,k]= t 

    #============================================================ 
    with record_function("my_rr_rayboxinstr"):
        
        #if 1:
        instr_list=[]
        instr_tNear_list=[]
        instr_tFar_list=[]
        
        # K,C 
        aabb_min=obj_bounds_world[:,0,:]
        aabb_max=obj_bounds_world[:,1,:]

        # * skip ray_dir normalization 
        # frame, rays, object_space , xyz  
        # F,N,K,C

        for i in range(f_num):
            # N,K,C
            i_instr, i_tnear, i_tfar=ray_box_intersection(
                                        obj_rays_o[i],
                                        obj_rays_d[i],
                                        aabb_min,
                                        aabb_max, 
                                        return_t=True )

            instr_list.append(i_instr)
            instr_tNear_list.append(i_tnear)
            instr_tFar_list.append(i_tfar)

    #---------------- 
    obj_instr  = torch.stack(instr_list,dim=0)
    obj_instr  = rearrange(obj_instr,'f n k -> k f n')
    
    obj_instr_tNear  = torch.stack(instr_tNear_list,dim=0)
    obj_instr_tNear  = rearrange(obj_instr_tNear,'f n k -> k f n')

    obj_instr_tFar  = torch.stack(instr_tFar_list,dim=0)
    obj_instr_tFar  = rearrange(obj_instr_tFar,'f n k -> k f n')

    obj_rays_o = rearrange(obj_rays_o , 'f n k c -> k f n c')
    obj_rays_d = rearrange(obj_rays_d , 'f n k c -> k f n c')
    obj_rays_ti= rearrange(obj_rays_ti, 'f n k   -> k f n')

    obj_rays_val = obj_instr * (obj_rays_ti!=-1)


    K,F,N = obj_instr.shape 
    M = render_sample_num

    # all_sigma = torch.zeros(K,F,N,M   ,device=device)
    all_sdf   = torch.ones(K,F,N,M   ,device=device)*10
    all_rgb   = torch.zeros(K,F,N,M,3 ,device=device)
    all_z     = torch.zeros(K,F,N,M   ,device=device)
    all_pts   = torch.zeros(K,F,N,M,3 ,device=device)
    all_nv    = torch.zeros(K,F,N,M,3 ,device=device)
    all_tkft  = torch.zeros(K,F,N,M,256 ,device=device)
    all_invs  = torch.ones(K,F,N,M   ,device=device)
    
    all_valid = torch.zeros(K,F,N,M       ,device=device,dtype=torch.bool)
    all_shift = torch.zeros(K,F,N,M,3     ,device=device)
    all_shift_valid = torch.zeros(K,F,N,M ,device=device,dtype=torch.bool)
    all_objvec= torch.zeros(K,F,N,M,K     ,device=device)
    

    with record_function("my_rr_searchZ"):
        
        for k in range(obj_num):
            all_objvec[k,:,:,:,k]=1.0
            
        for k in range(obj_num):
            motion_net = obj_kf_poses[k]

            valid = obj_rays_val[k]

            if valid.sum()==0: 
                if verbose:
                    print('[intersection] valid_sum=0')
                continue

            # rigid/world space 
            # x,c
            _rays_o  = obj_rays_o[k,  valid]
            _rays_d  = obj_rays_d[k,  valid]  
            _rays_ti = obj_rays_ti[k, valid]

            k_tnear=obj_instr_tNear[k, valid]
            k_tfar =obj_instr_tFar[k, valid]

            if motion_net.is_rigid(): 
                deform = False
            else: 
                deform = True
                
            rt = rays_search_z_single_obj(
                    ray_o=_rays_o,
                    ray_d=_rays_d,
                    ray_ti=_rays_ti, 
                    ray_tnear=k_tnear,
                    ray_tfar=k_tfar,
                    sample_mode=render_sample_mode, 
                    snum=render_sample_num,
                    sce_data=obj_sce_data_list[k],  
                    decoder=decoder, 
                    motion_net=motion_net,
                    deform=deform, 
                    near_z=near_z, 
                    far_z=far_z,
                    esti_normal=esti_normal,
                    render_tracker_ft=render_tracker_ft,
                    max_pre_samples=max_pre_samples,
                    device=device)

            if deform:
                #pred_shift = rt['p_shift'].reshape(-1,3)
                #all_shift_list.append(rt['p_shift'])
                all_shift[k,valid]=rt['p_shift']
                all_shift_valid[k,valid]=True

            # N, M, 
            #all_sigma[k, valid] = rt['sigma']
            all_sdf[k, valid]   = rt['sdf']
            all_rgb[k, valid]   = rt['rgb']
            all_z[k, valid]     = rt['z']
            all_pts[k,valid]    = rt['pts'] 
            all_valid[k,valid]  = rt['valid'] 
            all_invs[k,valid]   = rt['p_invs']

            #all_beta[k,valid]   = rt['beta']
            if esti_normal:
                all_nv[k,valid]    = rt['normal'] 
                #all_grad[k,valid]  = rt['p_grad']

            if render_tracker_ft:
                all_tkft[k,valid]  = rt['tracker_ft'] 


    #---------------------------------------------
    with record_function("my_rr_volumeRendering1"):

        #obj_all_z   = rearrange(all_z  , 'k f n m -> k (f n) m ').clone()
        #obj_all_invs= rearrange(all_invs,'k f n m -> k (f n) m ').clone()
        #obj_all_sdf = rearrange(all_sdf, 'k f n m -> k (f n) m ').clone()
        #obj_all_rgb = rearrange(all_rgb, 'k f n m c -> k (f n) m c').clone()
        obj_all_z   = all_z.clone()    
        obj_all_invs= all_invs.clone() 
        obj_all_sdf = all_sdf.clone()  
        obj_all_rgb = all_rgb.clone()  
        obj_all_valid = all_valid.clone() # K,F,N,M  

        # def volume_rendering(sdf, rgb, z_vals, invs, device, weight_only=False): 
        # k f n m
        _ort = volume_rendering(  sdf=obj_all_sdf,
                                  rgb=obj_all_rgb, 
                                  z_vals=obj_all_z,
                                  invs=obj_all_invs,
                                  valid=obj_all_valid,
                                  device=device )

        obj_depth, _, obj_color, obj_weights, obj_transmit = _ort 

    #--------------------------------------------- 
    with record_function("my_rr_post1"):

        all_z     = rearrange(all_z    ,'k f n m -> (f n) (k m)').clone()
        #all_sigma = rearrange(all_sigma,'k f n m -> (f n) (k m)').clone()
        all_sdf   = rearrange(all_sdf,  'k f n m -> (f n) (k m)').clone()
        all_invs  = rearrange(all_invs, 'k f n m -> (f n) (k m)').clone()
        all_rgb   = rearrange(all_rgb  ,'k f n m c -> (f n) (k m) c').clone()
        all_tkft  = rearrange(all_tkft  ,'k f n m c -> (f n) (k m) c').clone()
        all_pts   = rearrange(all_pts  ,'k f n m c -> (f n) (k m) c' ).clone()
        all_valid = rearrange(all_valid,'k f n m -> (f n) (k m) ' ).clone() 

        all_objvec = rearrange(all_objvec,'k f n m c -> (f n) (k m) c' ).clone()

        all_shift_valid  = rearrange(all_shift_valid ,
                                          'k f n m -> f n k m ' ).clone()
        all_shift  = rearrange(all_shift ,'k f n m c -> f n k m c' ).clone()

        #---------------------------------------------
        st_z, st_idx=all_z.sort(dim=-1)

        # (f n) (k m)
        # st_sigma = torch.gather(all_sigma, dim=-1,index=st_idx).clone()
        st_sdf   = torch.gather(all_sdf,   dim=-1, index=st_idx).clone()
        st_invs  = torch.gather(all_invs,  dim=-1, index=st_idx).clone()
        st_valid = torch.gather(all_valid, dim=-1, index=st_idx)
        
        _st_idx2 = repeat(st_idx,'fn km -> fn km c',c=3) 
        st_rgb   = torch.gather(all_rgb , dim=-2, index=_st_idx2).clone()
        st_pts   = torch.gather(all_pts , dim=-2, index=_st_idx2).clone()

        _st_idx3 = repeat(st_idx,'fn km -> fn km c', c=K) 
        st_objvec = torch.gather(all_objvec, dim=-2, index=_st_idx3).clone()

        _st_idx4 = repeat(st_idx,'fn km -> fn km c',c=all_tkft.shape[-1]) 
        st_tkft  = torch.gather(all_tkft, dim=-2, index=_st_idx4).clone()

    #----------------------------------------------
    with record_function("my_rr_volumeRendering2"):
        rt = volume_rendering(st_sdf, st_rgb, st_z,  st_invs, st_valid, device)

        pred_depth, uncertainty, pred_color, weights, transmit = rt 
        pred_depth  = rearrange(pred_depth  ,'(f n)   -> f n',   f=F, n=N) 
        pred_color  = rearrange(pred_color  ,'(f n) c -> f n c', f=F, n=N) 
        uncertainty = rearrange(uncertainty ,'(f n)   -> f n',   f=F, n=N) 

    #---------------------------------------------
    if esti_normal:

        with record_function("my_rr_esti_normal"):
            sh_st_idx = repeat(st_idx,'fn km -> fn km c',c=3) 
            all_nv  = rearrange(all_nv   ,'k f n m c -> (f n) (k m) c' ).clone()
            st_nv   = torch.gather(all_nv,  dim=-2, index=sh_st_idx).clone()

            wt_sum = weights.clone().sum(-1).unsqueeze(-1)
            wt_    = weights.clone().unsqueeze(-1)
            
            wt_sum = rearrange(wt_sum, '(f n) c -> f n c', f=F, n=N ) 
            pred_normal = reduce( wt_*st_nv[:,:-1,:], 
                            '(f n) km c -> f n 1 c', 'sum', f=F, n=N ) 

            #------------------
            sunlights = torch.tensor([[ 0.0000e+00, -9.9875e-01,  4.9938e-02],
                            [ 0.0000e+00, -1.0000e-03,  1.0000e+00],
                            [ 4.4721e-01, -8.9443e-04, -8.9443e-01],
                            [-4.4721e-01, -8.9443e-04, -8.9443e-01]]) 

            sunlights = sunlights.to(device)    

            sun_lm = torch.tensor([0.4,0.2,0.2,0.2], device=device)
            # 1 1 x
            sun_lm = sun_lm.reshape(1,1,-1) 

            sunlights = rearrange(sunlights, 'x c -> 1 1 x c')
            ambient=0.4
            # f n x 
            diffuse=(pred_normal*sunlights*-1).sum(dim=-1).clamp(min=0.0)        
            diffuse=(diffuse*sun_lm).sum(dim=-1)
            pred_shading = diffuse+ambient

            pred_shading = repeat(pred_shading,'f n -> f n c',c=3)
            pred_shading = pred_shading.clamp(min=0.0,max=1.0)

    else:
        pred_normal  = None 
        pred_shading = None
        pred_grad = None 
        grad_val =None

    #--------------------------------------------- 
    with record_function("my_rr_proc_output"):

        if render_tracker_ft:

            wt2_ = weights.unsqueeze(-1)
            tkft_map  = torch.sum(wt2_ * st_tkft[...,:-1,:],  -2 )  # (N_rays, 3) 
            
            pred_tkft = rearrange(tkft_map ,'(f n) c -> f n c', f=F, n=N) 

        else:
            pred_tkft = None

        #---------------------------------------------
        debug_train_pts = rearrange(all_pts  ,'(f n) (k m) c -> k f n m c',
                                    f=F, n=N, k=K).clone().detach()

        debug_valid     = rearrange(all_valid,'(f n) (k m) -> k f n m ',
                                    f=F, n=N, k=K).clone().detach()

        #---------------------------------------------  
        # st_valid2 = rearrange(st_valid, '(f n) (k m) -> k f n m '   ,f=F, n=N, k=K, m=M ).clone()
        # st_pts2   = rearrange(st_pts,'(f n) (k m) c -> k f n m c',f=F, n=N, k=K, m=M ).clone()
        
        #---------------------------------------------
        if render_obj_vec:

            # st_objvec do not have gradient.
            # (f n) km c
            pred_objvec = st_objvec[:,:-1] * (weights.clone().unsqueeze(-1))

            pred_objvec = reduce(pred_objvec, 
                                 '(f n) km c -> f n c', 'sum',  f=F, n=N)

        else:
            pred_objvec = None

    #--------------------------------------------- 

    out={
        'depth':pred_depth,
        'color':pred_color, 
        'tracker_ft':pred_tkft,
        'uncertainty':uncertainty,
        'objvec':pred_objvec,
        'normal':pred_normal,
        'shading':pred_shading,
        'obj_weights':obj_weights, 
        'obj_color'  :obj_color,
        'obj_depth'  :obj_depth,
        'obj_sample_z':obj_all_z,
        'p_shift'    :all_shift,
        'p_shift_valid':all_shift_valid,
        'debug_train_pts':debug_train_pts,
        'debug_valid':debug_valid,
        #'gradient':pred_grad,
        #'grad_val':grad_val,
        # 'obj_last_transmit':obj_last_trams,
        # 'obj_last_weight':obj_last_wt,
        #'sampled_w_pts':st_pts2,
        #'sampled_valid':st_valid2,
        #'alpha' :pred_obj_vis,
        'cano_obj_rays_o'  :obj_rays_o.clone(),     # k f n c
        'cano_obj_rays_d'  :obj_rays_d.clone(),     # k f n c
        'cano_obj_rays_valid':obj_rays_val.clone(), # k f n
    }

    return out 


