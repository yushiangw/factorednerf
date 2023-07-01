import numpy as np
import torch
import cv2 

def erode(val, tnum , erosion_size ):
    erosion_shape =  cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
    erosion_dst = cv2.erode(val, element, iterations=tnum)
    return erosion_dst
    
def dilate(val, tnum , dilatation_size ):
    dilation_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(val, element, iterations=tnum) 
    return dilatation_dst

def depth_filter_np(depth2, depth_sigma, space_sigma):
    
    mask = depth2==0
    
    depth2[mask]=-10*depth_sigma
    
    depth_flt = cv2.bilateralFilter(depth2,-1,depth_sigma,space_sigma)
    
    mask = mask.astype(np.float32)
    #mask=erode(mask,2, 4)
    mask=dilate(mask,1,2)
    mask=mask>0
    
    depth_flt[mask]=0
    return depth_flt, mask

def depth_filter(depth, depth_sigma=0.05, space_sigma=5):
    assert depth.ndim==2

    depth_np= depth.clone().cpu().squeeze().numpy()

    depth_np_fl,_=depth_filter_np(depth_np, depth_sigma=0.05, space_sigma=5)
    depth_fl= torch.from_numpy(depth_np_fl)
    depth_fl= depth_fl.to(torch.float32).to(depth.device)

    return depth_fl



def mask_erosion_np(mask_np ):
    assert mask_np.ndim==2

    #mask_np= mask.clone().cpu().numpy().astype(np.float32)
    
    #mask_np=dilate(mask_np,2, 4)
    mask_np=erode(mask_np, 1, 2)

    return mask_np

def mask_erosion(mask ):
    assert mask.ndim==2

    mask_np= mask.clone().cpu().numpy().astype(np.float32)
    

    mask2 = mask_erosion_np(mask_np)
    mask2 = torch.from_numpy(mask2).to(mask.dtype).to(mask.device)

    return mask2
