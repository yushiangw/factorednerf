import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import pdb
import pickle 
from scipy.spatial.transform import Rotation as R
import open3d as o3d 
from src.coord_tools import valid_backproj 


def unpack_mask(A,shape):
    return np.unpackbits(A, axis=None).reshape(shape).view(bool)
    

def o3d_makePcd(pts, color=None, rgb=None, normals=None): 
    assert pts.ndim==2
    
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        

    if normals is not None:
        _pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        
    if color is not None :
        _pcd.paint_uniform_color(color.reshape(3,1))
        
    if rgb is not None :
        _pcd.colors= o3d.utility.Vector3dVector(rgb)
    
    return _pcd


def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)


def is_rigid(mat):
    
    rot = mat[:3,:3]
    
    irot =rot @ (rot.T)
    
    valid1 = np.abs(irot - np.eye(3)).max() < 1.0e-7
    #if not valid1:
    #    print('err=',np.abs(irot - np.eye(3)).max())
        
    valid2 = np.abs(np.linalg.det(rot)-1) <1.0e-7
    #if not valid2:
    #    print('det=',np.abs(np.linalg.det(rot)-1))
        
    return valid1 and valid2
    

def enforce_rigid_np(mat):
    rot = mat[:3,:3]
    
    xyz  = R.from_matrix(rot).as_euler('xyz')
    #rot2=R.from_quat(qt).as_matrix()
    rot2 = R.from_euler('xyz', xyz).as_matrix()
    assert is_rigid(rot2)
    
    mat2=mat.copy()
    mat2[:3,:3]=rot2
    
    return mat2

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y



class BaseDataset(Dataset): 

    def __init__(self, cam_cfg, input_folder, device ):
        super().__init__()
        # cfg['cam']
        self.device = device
        #self.scale  = 1.0
        self.png_depth_scale = cam_cfg['png_depth_scale']

        self.H,  self.W  = cam_cfg['H'], cam_cfg['W']
        self.fx, self.fy = cam_cfg['fx'], cam_cfg['fy']
        self.cx, self.cy = cam_cfg['cx'], cam_cfg['cy']

        self.distortion = np.array( cam_cfg['distortion']) if 'distortion' in cam_cfg else None
        self.crop_size  = cam_cfg['crop_size'] if 'crop_size' in cam_cfg else None

        self.input_folder =  input_folder # cfg['data']['input_folder']

        self.crop_edge = cam_cfg['crop_edge']

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            #depth_data = readEXR_onlydepth(depth_path)
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) #*self.scale
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        #pose[:3, 3] *= self.scale
        return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)


class DYNSYNV3(BaseDataset):

    def __init__(self, db_cfg, cam_cfg, input_folder, device):
        super().__init__(cam_cfg, input_folder, device)
        
        data_fp = os.path.join(self.input_folder, 'data_v3.pkl')
        assert os.path.exists(data_fp),data_fp

        with open(data_fp,'rb') as fin:
            data = pickle.load(fin)  


        self.rgb           = data['rgba'][:,:,:,:3]
        self.segmentation  = data['segmentation'].squeeze(-1)

        self.depth         = data['depths_ns']
        #self.depth        = data['z'].squeeze(-1)
        self.normal        = data['esti_noise_nv']

        # self.normal = data['esti_nv']  

        #--------------------------------------------------------------
        _fp = os.path.join(self.input_folder, 'est_pose_v1_siammask.pkl')
        
        if os.path.exists(_fp):
            with open(_fp,'rb') as fin:
                x = pickle.load(fin) 
                #flow_poses_v1=x['flow_poses']
                obj_accum_poses = x['obj_accum_poses']
                est_obj_bbox = x['est_obj_bbox']
                gt_obj_bbox = x['gt_obj_bbox']
                del x 

            ebbox = np.stack(est_obj_bbox,axis=0)
            est_obj_bbox = torch.from_numpy(ebbox).to(torch.float32)

            gtbbox = np.stack(gt_obj_bbox,axis=0)
            gt_obj_bbox = torch.from_numpy(gtbbox).to(torch.float32)

            self.obj_accum_poses=obj_accum_poses
            self.est_obj_bbox=est_obj_bbox
            self.gt_obj_bbox=gt_obj_bbox
        else:
            self.obj_accum_poses=None

        #--------------------------------------------------------------
        tk_fp=os.path.join(self.input_folder,'siam_mask.pkl')
        if os.path.exists(tk_fp):
            with open(tk_fp,'rb') as fin:
                x = pickle.load(fin) 
                self.tk_seg=x['segmentation']

                self.tk_seg = torch.from_numpy(self.tk_seg)
 
                self.key_frame_idxs = x['key_frame_idxs']

                del x 
        else:
            self.tk_seg = None

        #--------------------------------------------------------------
        #(exr_layers["normal"].clip(-1.0, 1.0) + 1) * 65535 / 2
        #  ).astype(np.uint16)
        # h,w,c
        # (WORLD SPACE NORMAL)
        # _normal        = data['normal']        
        # self.normal = ((_normal).astype(np.float32)) *2/65535 -1 

        # Camera space normal 

        intrinsic         = data['intrinsic']
        my_pose_data      = data['my_pose_data']
        rigid_uid_list    = data['rigid_uid_list']
        nonrigid_uid_list = data['nonrigid_uid_list']

        #--------------------------------------------------------------

        self.my_pose_data = my_pose_data

        self.rigid_uid_list    = [0]+rigid_uid_list
        self.nonrigid_uid_list = nonrigid_uid_list
        self.all_dyn_uids      = rigid_uid_list+nonrigid_uid_list

        self.intrinsic = intrinsic 

        frame_num  = self.rgb.shape[0]
        self.n_img = frame_num

        self.cam_poses  = []
        self.objs_poses = []
        self.objs_to_world0 = []

        for i in range(frame_num):
            cam_c2w = my_pose_data[i]['cam_cam2world']
            cam_c2w = torch.from_numpy(cam_c2w).to(torch.float32) 

            objs_w2wobj0 = my_pose_data[i]['objs_obj2head']

            pd={}
            pd[0]=cam_c2w

            wpd={}
            wpd[0]=torch.eye(4)

            for uid in self.all_dyn_uids:

                _w2wobj0 = objs_w2wobj0[uid] 

                if not is_rigid(_w2wobj0):
                    _w2wobj0 = enforce_rigid_np(_w2wobj0)
                
                w2wobj0 = torch.from_numpy(_w2wobj0).to(torch.float32) 

                # w to w0 @ cam to w = cam to w0
                obj_pose = w2wobj0 @ cam_c2w
                
                pd[uid] =obj_pose  
                wpd[uid]=w2wobj0

            self.objs_poses.append(pd)
            self.objs_to_world0.append(wpd)
            self.cam_poses.append(cam_c2w)

        # for legacy BaseClass
        self.poses = [ torch.eye(4,device=device)  for i in range(len(my_pose_data))]
        
        #=============================================================

    def get_uids(self):

        return self.rigid_uid_list, self.nonrigid_uid_list

    def load_imgs(self, i):

        rgb =self.rgb[i]
        dep =self.depth[i]

        rgb = torch.from_numpy(rgb).to(torch.float32).to(self.device)
        rgb = rgb/255.0
        dep = torch.from_numpy(dep).to(torch.float32).to(self.device) 

        return rgb,dep

    def load_est_bbox(self):
        return self.est_obj_bbox.clone()

    def load_gt_bbox(self):
        return self.gt_obj_bbox.clone()

    def load_flow(self, i):

        fwflow =self.forward_flow[i]
        bkflow =self.backward_flow[i]

        fwflow = torch.from_numpy(fwflow).to(torch.float32).to(self.device)
        bkflow = torch.from_numpy(bkflow).to(torch.float32).to(self.device) 

        return fwflow,bkflow

    def load_normal(self, i):
        nv = self.normal[i]
        nv = torch.from_numpy(nv).to(torch.float32).to(self.device) 
        nv = F.normalize(nv, dim=-1)

        return nv 

    def load_seg(self, i ):
        seg=self.segmentation[i]
        seg=seg.astype(np.int16)
        seg=torch.from_numpy(seg).to(torch.int16)
        
        return seg 

    def load_siammask_seg(self, i ):
        assert self.tk_seg is not None

        seg=self.tk_seg[i].clone().detach().to(torch.int16)
        
        return seg 

    def load_est_poses_v1(self, i): 
        assert self.obj_accum_poses is not None
        
        objs_poses={}
        for uid in [0]+self.all_dyn_uids:
            
            mat44=self.obj_accum_poses[uid][i]
            mat44=torch.from_numpy(mat44).to(torch.float32).to(self.device) 
            objs_poses[uid] = mat44

        return objs_poses

    def load_my_poses(self, i): 
        #cam_pose  = self.cam_poses[i]
        objs_poses = self.objs_poses[i]

        return objs_poses

    def load_my_world_poses(self, i):

        #cam_pose  = self.cam_poses[i]
        objs_to_world0 = self.objs_to_world0[i]

        return objs_to_world0


class BeHave(BaseDataset):

    # data_fp='/ssd/BehavePKL_SSD/Date02_Sub02_tablesquare_move_k0/data.pkl'
    # seg_fp='/ssd/BehavePKL_SSD/Date02_Sub02_tablesquare_move_k0/siammask.pkl'
    # tk_save_fp='/ssd/BehavePKL_SSD/Date02_Sub02_tablesquare_move_k0/est_pose_v1_siammask.pkl'
    #
    #     'allf2keyf':allf2keyf,
    #     'keyframe_idxs':st_kidx,
    #     'all_rgbs_fp':all_rgbs_fp,
    #     'all_z_fp':all_z_fp,
    #     'key_seg':key_seg,
    #     'pc_table_ext':pc_table_ext,
    #     'local2world_R':local2world_R,
    #     'local2world_t':local2world_t,
    #     'scale':1000,
    # }

    def __init__(self, db_cfg, cam_cfg, input_folder, device):
        super().__init__(cam_cfg, input_folder, device)
        
        data_fp = os.path.join(self.input_folder, 'data.pkl')
        assert os.path.exists(data_fp), data_fp

        with open(data_fp,'rb') as fin:
            data = pickle.load(fin)   

        self.local2world_R=data['local2world_R']
        self.local2world_t=data['local2world_t']

        self.all_rgbs_fp = data['all_rgbs_fp']
        self.all_z_fp = data['all_z_fp']
        self.z_scale  = data['scale']

        assert self.z_scale==1000
        self.intrinsic = data['color_camera'] 
        #--------------------------------------------------------------
        seg_fp = os.path.join(self.input_folder, 'siammask.pkl')
        
        if os.path.exists(seg_fp):
            with open(seg_fp,'rb') as fin:
                seg_data = pickle.load(fin)
            
            # big array
            self.tk_obj_seg_list=seg_data['seg_mask_list']
            self.key_frame_idxs =seg_data['gt_fidx']
            self.tk_seg_uid_list=seg_data['uid_list']

            seg_fnum = len(self.tk_obj_seg_list[1])
            
            self.all_dyn_uids = self.tk_seg_uid_list
        else:
            seg_fnum = len( self.all_rgbs_fp )
            self.all_dyn_uids = [1,2]

        #-------------------------------------------------------------- 

        self.rigid_uid_list    = [0,1]
        self.nonrigid_uid_list = [2] 

        #--------------------------------------------------------------
        pose_save_fp = os.path.join(self.input_folder, 
                                'est_pose_v1_siammask.pkl')

        if os.path.exists(pose_save_fp):
            # out={
            #     'obj_accum_poses':obj_accum_poses,
            #     'icp_tx':sce_tx_list, 
            #     'est_obj_bbox':est_obj_bbox,
            #     'gt_obj_bbox':gt_bbox,

            with open(pose_save_fp,'rb') as fin:
                p_data = pickle.load(fin)

            obj_accum_poses = p_data['obj_accum_poses']
            est_obj_bbox = p_data['est_obj_bbox']
            gt_obj_bbox  = p_data['gt_obj_bbox']
            
            ebbox = np.stack(est_obj_bbox,axis=0)
            est_obj_bbox = torch.from_numpy(ebbox).to(torch.float32)

            gtbbox = np.stack(gt_obj_bbox,axis=0)
            gt_obj_bbox = torch.from_numpy(gtbbox).to(torch.float32)

            self.obj_accum_poses=obj_accum_poses
            self.est_obj_bbox=est_obj_bbox
            self.gt_obj_bbox=gt_obj_bbox

        
        #--------------------------------------------------------------
        self.n_img = seg_fnum

        self.save_mem=1
        #--------------------------------------------------------------
        # for legacy BaseClass
        self.poses = [ torch.eye(4,device=device) for i in range(self.n_img)]
        
        #=============================================================

    def get_uids(self):

        return self.rigid_uid_list, self.nonrigid_uid_list

    def load_imgs(self, i):

        rgb=cv2.imread(self.all_rgbs_fp[i])
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

        dep=cv2.imread(self.all_z_fp[i],cv2.IMREAD_ANYDEPTH)
        
        rgb = torch.from_numpy(rgb).to(torch.float32)/255.0
        dep  = torch.from_numpy(dep.astype(np.float32))
        dep  =dep.to(torch.float32)/self.z_scale

        if not self.save_mem:
            rgb = rgb.to(self.device) 
            dep = dep.to(self.device) 

        return rgb,dep

    def load_normal(self, i):
        # nv = self.normal[i]
        # nv = torch.from_numpy(nv).to(torch.float32).to(self.device) 
        # nv = F.normalize(nv, dim=-1)

        fx=self.intrinsic['fx']
        fy=self.intrinsic['fy']
        cx=self.intrinsic['cx']
        cy=self.intrinsic['cy']
        H =self.intrinsic['height']
        W =self.intrinsic['width']

        rgb,dep = self.load_imgs(i)

        if self.save_mem:
            c2w = torch.eye(4)
        else:
            c2w = torch.eye(4).to(self.device)

        pcd, pcd_rgb, mask =valid_backproj(c2w, rgb, dep,
                                           0, H, 0, W, 
                                           fx, fy, cx, cy,
                                           return_img=False)


        pcd=pcd.cpu().numpy()
        pcd_o=o3d_makePcd(pcd)
        pcd_o.estimate_normals()
        pcd_o.orient_normals_towards_camera_location(np.zeros((3,)))
        pcd_o.normalize_normals()

        cam_nv = np.asarray(pcd_o.normals) 

        esti_nv = np.zeros((H,W,3))
        esti_nv[mask.cpu().numpy()]=cam_nv 

        nv = torch.from_numpy(esti_nv).to(torch.float32)

        if not self.save_mem:
            nv=nv.to(self.device) 

        return nv 

    def load_est_bbox(self):
        return self.est_obj_bbox.clone()

    def load_gt_bbox(self):
        return self.gt_obj_bbox.clone()

    def load_seg(self, i ):
        #seg=self.segmentation[i]
        #seg=seg.astype(np.int16)
        #seg=torch.from_numpy(seg).to(torch.int16)
        raise Exception('not support')
        return seg 

    def load_siammask_seg(self, i ):
        assert self.tk_obj_seg_list is not None

        H,W = self.tk_obj_seg_list[1][0].shape

        seg = torch.zeros((H,W),dtype=torch.int16)

        for uid in self.tk_seg_uid_list: 

            i_mask = self.tk_obj_seg_list[uid][i]

            seg[i_mask>0]=uid

        return seg 

    def load_est_poses_v1(self, i): 
        assert self.obj_accum_poses is not None
        
        objs_poses={}

        # static BG
        objs_poses[0]=torch.eye(4,device=self.device)

        for uid in self.all_dyn_uids:
            
            mat44=self.obj_accum_poses[uid][i]
            mat44=torch.from_numpy(mat44).to(torch.float32).to(self.device) 
            objs_poses[uid] = mat44

        return objs_poses

    def load_my_poses(self, i): 
        raise Exception('not support')
        #cam_pose  = self.cam_poses[i]
        objs_poses = self.objs_poses[i]

        return objs_poses

    def load_my_world_poses(self, i):
        raise Exception('not support') 

        #cam_pose  = self.cam_poses[i]
        objs_to_world0 = self.objs_to_world0[i]

        return objs_to_world0



class BeHave_v2(BaseDataset):
    # 
    #     'allf2keyf':,
    #     'keyframe_idxs':,
    #
    #     'all_rgbs_fp': []
    #     'all_z_fp': [] 
    #
    def __init__(self, db_cfg, cam_cfg, input_folder,  device):

        super().__init__(cam_cfg, input_folder, device)
        
        self.raw_db_root=db_cfg['raw_db_root']

        data_fp = os.path.join(self.input_folder, 'data.pkl')
        assert os.path.exists(data_fp), data_fp

        with open(data_fp,'rb') as fin:
            data = pickle.load(fin)   

        self.anno_keyframe_idxs=data['keyframe_idxs']

        self.local2world_R=data['local2world_R']
        self.local2world_t=data['local2world_t']

        self.all_rgbs_fp = data['all_rgbs_fp']
        self.all_z_fp = data['all_z_fp']
        self.z_scale  = data['scale']

        assert self.z_scale==1000
        self.intrinsic = data['color_camera'] 

        #--------------------------------------------------------------
        seg_fp = os.path.join(self.input_folder, 'siammask.pkl')
        
        if os.path.exists(seg_fp):
            with open(seg_fp,'rb') as fin:
                seg_data = pickle.load(fin)

            # all_masks: {uid}{fidx}
            # tk_obj_seg_list
            self.all_masks=seg_data['all_masks']
            #seg_fnum = len(self.all_masks[1])

            self.mask_h =  seg_data['im_h']
            self.mask_w = seg_data['im_w']

            self.key_frame_idxs =seg_data['gt_fidx']
            self.seg_uid_list=seg_data['uid_list']

            
            self.all_dyn_uids = self.seg_uid_list
        else:
            #seg_fnum = len( self.all_rgbs_fp )
            self.all_dyn_uids = [1,2]

        #-------------------------------------------------------------- 

        self.rigid_uid_list    = [0,1]
        self.nonrigid_uid_list = [2] 

        #--------------------------------------------------------------
        bbox_fp = os.path.join(self.input_folder, 'bbox.pkl')

        pose_save_fp = os.path.join(self.input_folder, 
                                    'est_pose_v2_siammask.pkl')

        if os.path.exists(pose_save_fp):
            # training mode

            with open(pose_save_fp,'rb') as fin:
                tk = pickle.load(fin)

            # {uid}{fidx}
            self.obj_accum_poses = tk['obj_accum_poses'] 

            all_fidxs=list(tk['obj_accum_poses'][0].keys())
            all_fidxs.sort()
            self.all_fidxs = all_fidxs

            self.start_fidx = tk['start_fidx']
            #------------------------------------
            with open(bbox_fp,'rb') as fin:
                bbox=pickle.load(fin) 
                
            gt_obj_bbox = bbox['gt_obj_bbox']
            gt_obj_bbox = np.asarray(gt_obj_bbox)
            self.gt_obj_bbox = torch.from_numpy(gt_obj_bbox).to(torch.float32) 
        else:
            self.start_fidx = 0

            self.all_fidxs = [i for i in range( len( self.all_rgbs_fp )) ]

        #--------------------------------------------------------------
        self.n_img = len(self.all_fidxs)

        self.save_mem=1

        #--------------------------------------------------------------
        # for legacy BaseClass
        self.poses = [ torch.eye(4,device=device) for i in range(self.n_img)]
        
        #=============================================================

    def get_uids(self): 
        return self.rigid_uid_list, self.nonrigid_uid_list

    def load_imgs(self, i): 
        real_idx = self.start_fidx + i

        rgb_fp = os.path.join(self.raw_db_root, self.all_rgbs_fp[real_idx])
        z_fp   = os.path.join(self.raw_db_root, self.all_z_fp[real_idx])

        assert os.path.exists(rgb_fp), rgb_fp
        
        rgb=cv2.imread(rgb_fp)
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

        dep=cv2.imread(z_fp,cv2.IMREAD_ANYDEPTH)
        
        rgb = torch.from_numpy(rgb).to(torch.float32)/255.0
        dep  = torch.from_numpy(dep.astype(np.float32))
        dep  =dep.to(torch.float32)/self.z_scale

        if not self.save_mem:
            rgb = rgb.to(self.device) 
            dep = dep.to(self.device) 

        return rgb,dep

    def load_normal(self, i):
        # nv = self.normal[i]
        # nv = torch.from_numpy(nv).to(torch.float32).to(self.device) 
        # nv = F.normalize(nv, dim=-1)

        rgb,dep = self.load_imgs(i=i)

        fx=self.intrinsic['fx']
        fy=self.intrinsic['fy']
        cx=self.intrinsic['cx']
        cy=self.intrinsic['cy']
        H =self.intrinsic['height']
        W =self.intrinsic['width']


        if self.save_mem:
            c2w = torch.eye(4)
        else:
            c2w = torch.eye(4).to(self.device)

        pcd, pcd_rgb, mask =valid_backproj(c2w, rgb, dep,
                                           0, H, 0, W, 
                                           fx, fy, cx, cy,
                                           return_img=False)


        pcd=pcd.cpu().numpy()
        pcd_o=o3d_makePcd(pcd)
        pcd_o.estimate_normals()
        pcd_o.orient_normals_towards_camera_location(np.zeros((3,)))
        pcd_o.normalize_normals()

        cam_nv = np.asarray(pcd_o.normals) 

        esti_nv = np.zeros((H,W,3))
        esti_nv[mask.cpu().numpy()]=cam_nv 

        nv = torch.from_numpy(esti_nv).to(torch.float32)

        if not self.save_mem:
            nv=nv.to(self.device) 

        return nv 

    def load_seg(self, i ):
        #seg=self.segmentation[i]
        #seg=seg.astype(np.int16)
        #seg=torch.from_numpy(seg).to(torch.int16)
        raise Exception('not support')
        return seg 

    def load_est_bbox(self): 
        raise Exception('not support')
        return seg 

    def load_gt_bbox(self):
        return self.gt_obj_bbox.clone()


    def load_siammask_seg(self, i ):
        assert self.all_masks is not None

        real_idx = self.start_fidx + i

        H =self.intrinsic['height']
        W =self.intrinsic['width']

        seg = torch.zeros((H,W),dtype=torch.int16)

        for uid in self.all_dyn_uids: 

            _mask = self.all_masks[uid][real_idx]
            _mask = unpack_mask(_mask, 
                         (self.mask_h, self.mask_w))

            _mask = _mask.astype(np.float32)

            _mask = cv2.resize(_mask,(W,H),cv2.INTER_NEAREST)
            _mask = _mask>0  

            seg[_mask>0]=uid

        return seg 

    def load_est_poses_v1(self, i): 
        assert self.obj_accum_poses is not None
        
        real_idx = self.start_fidx + i

        objs_poses={}

        # static BG
        objs_poses[0]=torch.eye(4,device=self.device)

        for uid in self.all_dyn_uids:
            
            mat44=self.obj_accum_poses[uid][real_idx]
            mat44=torch.from_numpy(mat44).to(torch.float32).to(self.device) 
            objs_poses[uid] = mat44

        return objs_poses

    def load_my_poses(self, i): 
        raise Exception('not support')

    def load_my_world_poses(self, i):
        raise Exception('not support') 