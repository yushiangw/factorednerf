
import torch 
import numpy as np
from einops import rearrange, reduce, repeat 
import pdb 

def ray_box_intersection(ray_o, ray_ud, aabb_min, aabb_max, return_t=False):
    """
    Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/main/neural_scene_graph_helper.py
    Args:
        ray_o:  [rays, boxes, 3]
        ray_ud: [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns: intersected [rays, boxes]
    
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    
    #if aabb_min is None:
    #    aabb_min = torch.ones_like(ray_o) * -1. # constant([-1., -1., -1.])
    #if aabb_max is None:
    #    aabb_max = torch.ones_like(ray_o) # constant([1., 1., 1.])
    
    #torch.autograd.set_detect_anomaly(False)

    assert aabb_min.ndim==2
    assert aabb_max.ndim==2
    
    #try:
    # N,B,3
    aabb_min = rearrange(aabb_min, 'b c -> 1 b c')
    aabb_max = rearrange(aabb_max, 'b c -> 1 b c')

    #------------------------------------------------
    # ray_ud_g[vmask] = ray_ud[vmask]
    # out = 1/input 

    # if 0:
    #     # N,B,3
    #     vmask = (ray_ud.abs()>0)

    #     # N,B
    #     vmask_sum = (vmask.sum(dim=-1)==3)
    #     inv_d = torch.ones_like(ray_ud)*9999999*torch.sign(ray_ud)
    #     inv_d[vmask]= torch.reciprocal(ray_ud[vmask])


    inv_d = torch.reciprocal(ray_ud) 
    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    # N,B,3
    t0 = torch.fmin(t_min, t_max)
    t1 = torch.fmax(t_min, t_max)
    
    # N,B,
    t_near = torch.fmax(torch.fmax(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far  = torch.fmin(torch.fmin(t1[..., 0], t1[..., 1]), t1[..., 2])
    
    # return max_component(tmin) <= min_component(tmax);
    # return t_near <= t_far
    # return (t_near <= t_far) * (t_far>0)     
    intersected  = (t_near <= t_far) * (t_far>0) 
    
    if return_t:
        return intersected, t_near, t_far
    else:
        return intersected 
    #
    # except:
    #     pdb.set_trace()
