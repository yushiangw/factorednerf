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
    # def __init__(self, cfg, args, scale, device='cuda:0'  ):

    def __init__(self, cfg, scale, device='cuda:0'  ):
        super(BaseDataset, self).__init__()
    
        self.name   = cfg['dataset']
        self.device = device
        self.scale  = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array( cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        self.input_folder = cfg['data']['input_folder']

        self.crop_edge = cfg['cam']['crop_edge']

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
        depth_data = torch.from_numpy(depth_data)*self.scale
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
        pose[:3, 3] *= self.scale
        return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)


class Replica(BaseDataset):
    def __init__(self, cfg, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, cfg, scale, device='cuda:0'
                 ):
        super(Azure, self).__init__(cfg, scale, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(
            self.input_folder, 'scene', 'trajectory.log'))

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, scale, device)
        self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg,  scale, device='cuda:0'  ):
        super(CoFusion, self).__init__(cfg,  scale, device)
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth_noise', '*.exr')))
        self.n_img = len(self.color_paths)
        
        #self.load_poses(os.path.join(self.input_folder, 'trajectories'))
        
        self.my_cam_poses, tids=self.load_poses_v2(os.path.join(self.input_folder, 'trajectories/gt-cam-0.txt'))
        self.my_obj1_poses, _  =self.load_poses_v2(os.path.join(self.input_folder, 'trajectories/gt-ship-1.txt'))
        self.my_obj2_poses, _  =self.load_poses_v2(os.path.join(self.input_folder, 'trajectories/gt-car-2.txt'))
        self.my_obj3_poses, _  =self.load_poses_v2(os.path.join(self.input_folder, 'trajectories/gt-poses-horse-3.txt'))

        self.poses = []
        for t in tids:
            pm=self.my_cam_poses[t]
            self.poses.append(pm)

        self.mask_paths = []
        for f in self.color_paths:
            f2=f.replace('colour/Color','mask_id/Mask')
            self.mask_paths.append(f2)

    def load_mask(self, index):
        _path = self.mask_paths[index] 
        mask= cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
        #cv2.imread(_path, cv2.IMREAD_UNCHANGED)
        mask=torch.from_numpy(mask).to(self.device)
        return mask

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, 3] = pvec[:3]
        pose[:3, :3]= Rotation.from_quat(pvec[3:]).as_matrix()
        return pose

    def load_poses_v2(self, path):

        with open(path, "r") as f:
            lines = f.readlines()

        inv_pose0 = None
        num = len(lines)

        poses = {} 
        tids  = []
        for i in range(num):
            line = lines[i]
            pvec = np.array(list(map(float, line.split()))).reshape(-1)
            
            # ts 
            tid  = int(pvec[0])
            tid  = tid-1 # cf start from 1 

            pvec = pvec[1:]

            c2w = self.pose_matrix_from_quaternion(pvec)

            if inv_pose0 is None:
                inv_pose0 = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose0@c2w
            
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1 
            c2w = torch.from_numpy(c2w).to(torch.float32).to(self.device) 
            
            poses[tid]=c2w
            tids.append(tid)

        return poses,tids

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy.
        # But it will not affect the calculation of ATE since camera trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg,  scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg,  scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class DYNSYN(BaseDataset):
    def __init__(self, cfg, scale, device='cuda:0'
                 ):
        super().__init__(cfg,  scale, device)

        # /DynSYNTH/fullrd_000217/rgb/0000.00000.png
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/rgb/*.00000.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/depth/*.00000.png'))

        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/result_nogt.txt')

        # self.png_depth_scale = cfg['cam']['png_depth_scale']

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, 3] = pvec[:3]
        pose[:3, :3]= Rotation.from_quat(pvec[3:]).as_matrix()
        return pose

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        inv_pose = None

        for i in range(self.n_img):
            line = lines[i]
            pvec = np.array(list(map(float, line.split()))).reshape(-1)
            # ts 
            pvec = pvec[1:]

            c2w = self.pose_matrix_from_quaternion(pvec)

            #if inv_pose is None:
            #    inv_pose = np.linalg.inv(c2w)
            #    c2w = np.eye(4)
            #else:
            #    c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            c2w = torch.from_numpy(c2w).to(torch.float32)

            self.poses.append(c2w)


class RFScan(BaseDataset):
    def __init__(self, cfg,  scale, device='cuda:0'
                 ):
        super().__init__(cfg,  scale, device)

        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/rgb/*.00000.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/depth/*.00000.png'))

        self.n_img = len(self.color_paths)
        
        # self.load_poses(f'{self.input_folder}/RF_result.txt')
        self._ini_iden_pose()

        # self.png_depth_scale = cfg['cam']['png_depth_scale']

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, 3] = pvec[:3]
        pose[:3, :3]= Rotation.from_quat(pvec[3:]).as_matrix()
        return pose

    def _ini_iden_pose(self):
        self.poses = []  

        for i in range(self.n_img):
            c2w = torch.eye(4).to(torch.float32)

            self.poses.append(c2w)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        inv_pose = None

        for i in range(self.n_img):
            line = lines[i]
            pvec = np.array(list(map(float, line.split()))).reshape(-1)
            # ts 
            pvec = pvec[1:]

            c2w = self.pose_matrix_from_quaternion(pvec)

            #if inv_pose is None:
            #    inv_pose = np.linalg.inv(c2w)
            #    c2w = np.eye(4)
            #else:
            #    c2w = inv_pose@c2w
            
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            c2w = torch.from_numpy(c2w).to(torch.float32)

            self.poses.append(c2w)


class DYNSYNV2(BaseDataset):

    def __init__(self, cfg,  scale, device='cuda:0' ):
        super().__init__(cfg,  scale, device)
        
        meta_fp = os.path.join(self.input_folder, 'my_pose_data.pkl')

        with open(meta_fp,'rb') as fin:
            data = pickle.load(fin) 
        meta = data['meta']

        self.meta = meta 

        my_pose_data = meta['my_pose_data']  

        FNUM = len(my_pose_data)

        self.cam_poses  = []
        self.objs_poses = []

        for i in range(FNUM):
            cam_c2w = my_pose_data[i]['cam_cam2world']
            cam_c2w = torch.from_numpy(cam_c2w).to(torch.float32) 

            objs_w2wobj0 = my_pose_data[i]['objs_obj2head']

            pd = []
            for j in range(len(objs_w2wobj0)):
                w2wobj0 = objs_w2wobj0[j]
                w2wobj0 = torch.from_numpy(w2wobj0).to(torch.float32) 

                # c2w
                obj_pose = w2wobj0 @ cam_c2w 

                pd.append(obj_pose)

            self.objs_poses.append(pd)
            self.cam_poses.append(cam_c2w)

        self.poses = [ torch.eye(4,device=device)  for i in range(len(my_pose_data))]
        
        #=============================================================

        _paths = glob.glob(f'{self.input_folder}/render/rgb_*.jpg')

        self.color_paths = [ os.path.join(self.input_folder, f'render/rgb_{i:04d}.jpg')   for i in range(FNUM) ] 
        self.seg_paths   = [ os.path.join(self.input_folder, f'render/seg_{i:04d}.png')   for i in range(FNUM) ]  
        self.depth_paths = [ os.path.join(self.input_folder, f'render/depth_{i:04d}.exr') for i in range(FNUM) ]  

        self.n_img = len(self.color_paths) 

    def load_seg(self, i ):

        seg_fp = self.seg_paths[i]
        seg=cv2.imread(seg_fp,cv2.IMREAD_ANYDEPTH).astype(np.int16)
        seg=torch.from_numpy(seg).to(torch.int16)
        
        return seg 

    def load_my_poses(self, i):

        cam_pose  = self.cam_poses[i]
        objs_pose = self.objs_poses[i]

        return cam_pose, objs_pose


class DYNSYNV3(BaseDataset):

    def __init__(self, cfg,  scale, device='cuda:0' ):
        super().__init__(cfg,  scale, device)
        
        data_fp = os.path.join(self.input_folder, 'data.pkl')
        assert os.path.exists(data_fp)

        with open(data_fp,'rb') as fin:
            data = pickle.load(fin)  

        self.rgb           = data['rgba'][:,:,:,:3]
        self.forward_flow  = data['forward_flow']
        self.backward_flow = data['backward_flow']
        self.depth         = data['z'].squeeze(-1)
        self.segmentation  = data['segmentation'].squeeze(-1)
        self.normal        = data['normal']

        intrinsic         = data['intrinsic']
        my_pose_data      = data['my_pose_data']
        rigid_uid_list    = data['rigid_uid_list']
        nonrigid_uid_list = data['nonrigid_uid_list']
        #nonrigid_uidxpartid_list= data['nonrigid_uidxpartid_list']

        self.my_pose_data = my_pose_data

        self.rigid_uid_list = [0]+rigid_uid_list
        self.nonrigid_uid_list = nonrigid_uid_list
        self.all_dyn_uids = rigid_uid_list+nonrigid_uid_list

        self.intrinsic = intrinsic 

        frame_num = self.rgb.shape[0]
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

                w2wobj0 = objs_w2wobj0[uid] 
                w2wobj0 = enforce_rigid_np(w2wobj0)
                w2wobj0 = torch.from_numpy(w2wobj0).to(torch.float32) 

                # w to w0 @ cam to w = cam to w0
                obj_pose = w2wobj0 @ cam_c2w  

                
                pd[uid]=obj_pose 

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

    def load_flow(self, i):

        fwflow =self.forward_flow[i]
        bkflow =self.backward_flow[i]

        fwflow = torch.from_numpy(fwflow).to(torch.float32).to(self.device)
        bkflow = torch.from_numpy(bkflow).to(torch.float32).to(self.device) 

        return fwflow,bkflow

    def load_seg(self, i ):

        seg=self.segmentation[i]
        seg=seg.astype(np.int16)
        seg=torch.from_numpy(seg).to(torch.int16)
        
        return seg 

    def load_my_poses(self, i):

        #cam_pose  = self.cam_poses[i]
        objs_poses = self.objs_poses[i]

        return objs_poses

    def load_my_world_poses(self, i):

        #cam_pose  = self.cam_poses[i]
        objs_to_world0 = self.objs_to_world0[i]

        return objs_to_world0



dataset_dict = {
    "replica":  Replica,
    "scannet":  ScanNet,
    "cofusion": CoFusion,
    "azure":    Azure,
    "tumrgbd":  TUM_RGBD,
    "dynsyn":   DYNSYN,
    "rfscan":   RFScan,
    'dynsyn2':  DYNSYNV2,
    'dynsyn3':  DYNSYNV3,
}
