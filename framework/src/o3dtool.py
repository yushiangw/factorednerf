import open3d as o3d
import numpy as np
import copy
import trimesh 
import pdb
#import cv2
from scipy.spatial.transform import Rotation as R 


def makePcd(pts, color=None, rgb=None, normals=None): 
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


def makeSphere(center, radius, color=None ):  
    
    sph = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=8)
    sph = sph.translate(center.reshape(3,1))

    if color is not None :
        sph.paint_uniform_color(color.reshape(3,1))
    
    return sph


def makeBox2(min,max):

    vex=[]
    vex.append((min[0],min[1],min[2]))
    vex.append((max[0],max[1],max[2]))

    vex=np.asarray(vex)

    idx=[]
    idx.append([0,1]) 

    line = o3d.geometry.LineSet()
    line.points=o3d.utility.Vector3dVector( vex )
    line.lines =o3d.utility.Vector2iVector( idx )

    return line



def makeBox(min,max):

    vex=[]
    vex.append((min[0],min[1],min[2]))
    vex.append((min[0],min[1],max[2]))
    vex.append((max[0],min[1],max[2]))
    vex.append((max[0],min[1],min[2]))

    vex.append((min[0],max[1],min[2]))
    vex.append((min[0],max[1],max[2]))
    vex.append((max[0],max[1],max[2]))
    vex.append((max[0],max[1],min[2]))

    vex=np.asarray(vex)

    idx=[]
    idx.append([0,1])
    idx.append([1,2])
    idx.append([2,3])
    idx.append([3,0])

    idx.append([4,5])
    idx.append([5,6])
    idx.append([6,7])
    idx.append([7,4])

    idx.append([0,4])
    idx.append([1,5])
    idx.append([2,6])
    idx.append([3,7])

    line = o3d.geometry.LineSet()
    line.points=o3d.utility.Vector3dVector( vex )
    line.lines =o3d.utility.Vector2iVector( idx )

    return line

def makePlane(minv,maxv,y=None):

    if y is None:
        y=minv 

    vex=[]
    vex.append((minv,y,minv))
    vex.append((minv,y,maxv))
    vex.append((maxv,y,maxv))
    vex.append((maxv,y,minv)) 
    vex=np.array(vex)

    idx=[]
    idx.append([0,1,2])
    idx.append([2,3,0])

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices=o3d.utility.Vector3dVector(  vex )
    mesh_o3d.triangles=o3d.utility.Vector3iVector( idx)

    return mesh_o3d


def makeMesh( vertices_n3, faces_n3, normals_n3=None, color=None, vexcolor=None):

    mesh_o3d= o3d.geometry.TriangleMesh()
    
    mesh_o3d.vertices =o3d.utility.Vector3dVector( vertices_n3 )
    mesh_o3d.triangles=o3d.utility.Vector3iVector( faces_n3)

    if normals_n3 is not None:
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector( normals_n3 )
    else:
        # for shading 
        mesh_o3d.compute_triangle_normals()
        mesh_o3d.compute_vertex_normals()
    
    if vexcolor is not None:
        mesh_o3d.vertex_colors =o3d.utility.Vector3dVector( vexcolor )

    if color is not None:
        mesh_o3d.paint_uniform_color(color) 
    
    return mesh_o3d



def makeLineMesh(mesh_o3d):
    line_mesh_o3d=o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)
    
#def geto3d_cams(fp):
#    mesh = o3d.io.read_triangle_mesh(fp)
def load_obj(fp):
    mesh = trimesh.load_mesh(fp)
    if not isinstance(mesh,trimesh.Trimesh):
        #  m.visual = trimesh.visual.ColorVisuals()
        mesh_dump=mesh.dump()
        for x in mesh_dump:
            x.visual = trimesh.visual.ColorVisuals()
        mesh=mesh_dump.sum()
        
    mesh_o3d= o3d.geometry.TriangleMesh()
    
    mesh_o3d.vertices=o3d.utility.Vector3dVector( mesh.vertices )
    mesh_o3d.triangles=o3d.utility.Vector3iVector( mesh.faces)
    mesh_o3d.paint_uniform_color((0.7,0.7, 0.8))
    mesh_o3d.compute_triangle_normals()
    mesh_o3d.compute_vertex_normals()
    
    return mesh_o3d

class O3DVisWrap: 
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.w=640*2
        self.h=480*2
        self.vis.create_window(width=self.w, height=self.h, left=5, top=5)
        
        self.ropt=self.vis.get_render_option()
        self.vc  =self.vis.get_view_control()
        self.vc.set_zoom(1.0)
        
        #self.vc.rotate(800.0, 100.0) 
        #self.opt.line_width=1
        
    def close(self):
        self.vis.destroy_window()
    
    def __del__(self):
        self.close()
    
    def get_current_ext(self):
        vc=self.vis.get_view_control()
        cam=vc.convert_to_pinhole_camera_parameters()
        
        return np.asarray(cam.extrinsic)
        
        
    def render_pose(self, mesh_list_in, pose, scale=1.0 , ext=None):
        
        mesh_list=[]
        
        for m in mesh_list_in:
            mesh_list.append( copy.deepcopy(m) )
            
        for m in mesh_list:
            m.transform(pose)
            self.vis.add_geometry(m) 
        
        vc  =self.vis.get_view_control()
        #vc.scale(scale)
        cam=vc.convert_to_pinhole_camera_parameters()
        
        if ext is not None:
            cam.extrinsic=ext
            vc.convert_from_pinhole_camera_parameters(cam)
            
        self.vis.update_renderer() 
        
        img1=self.vis.capture_screen_float_buffer(do_render=True)
        img1=np.asarray(img1)
        
        if img1.shape[0] < self.h or img1.shape[1] < self.w:
            #3 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img2 = np.zeros((self.h, self.w,3),dtype=np.uint8)
            img2[:img1.shape[0],:img1.shape[1],:]=img1
            img1=img2 

        for m in mesh_list:
            self.vis.remove_geometry(m) 
            
        self.vis.poll_events()
        self.vis.update_renderer()
        
        return img1
    
    
    def render_pose_list(self, mesh_list_in, pose_list, scale=1.0 , ext=None):
        
        num=len(pose_list)
        img_list=[] 
        
        for k in range(num): 
            k_pose=pose_list[k]
            _im=self.render_pose( mesh_list_in, pose=k_pose, scale=scale , ext=ext) 
            img_list.append(_im)
        
        if len(img_list)>0:
            img_cat=np.concatenate(img_list,axis=1)
        else:
            img_cat=img_list[0]
            
        return img_cat
  

def get_render_poses():
    pose_list=[] 

    for deg in [0,180,90,60,45,15]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('y', (180+deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [90,60,30]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (180,deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [30,60,90]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (270,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 
    
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                 [-0.,         -1.,         -0.,    0.1  ],
                 [-0. ,        -0.,         -1.,    2  ],
                 [ 0.  ,        0.,          0.,    1.0  ]])


    return pose_list, cam_ext

def get_render_poses2():
    pose_list=[] 

    for deg in [0,80,160]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('y', (180+deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [60]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (180,deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [60]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (280,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 
    
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                 [-0.,         -1.,         -0.,    0.1  ],
                 [-0. ,        -0.,         -1.,    2  ],
                 [ 0.  ,        0.,          0.,    1.0  ]])


    return pose_list, cam_ext

def get_render_poses3():
    pose_list=[] 

    for deg in [0]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('y', (180+deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [30,60,90]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (180,deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                 [-0.,         -1.,         -0.,    0.1  ],
                 [-0. ,        -0.,         -1.,    2  ],
                 [ 0.  ,        0.,          0.,    1.0  ]])


    return pose_list, cam_ext

def get_render_poses4():
    pose_list=[] 

    for deg in [0]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('y', (180+deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [60]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (180,deg), degrees=True).as_matrix() 
        pose_list.append(p) 
        
    for deg in [45]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (240,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 
     
    for deg in [-45]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (10,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 
    
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                      [-0.,     -1.,    -0.,    0.1  ],
                      [-0.,     -0.,    -1.,    2    ],
                      [ 0.,      0.,     0.,    1.0  ]])

    return pose_list, cam_ext


def get_render_poses5():
    pose_list=[] 

    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (160,60), degrees=True).as_matrix() 
    pose_list.append(p) 
 
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (240,-45), degrees=True).as_matrix() 
    pose_list.append(p) 
 
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                 [-0.,         -1.,         -0.,    0.1  ],
                 [-0. ,        -0.,         -1.,    2  ],
                 [ 0.  ,        0.,          0.,    1.0  ]])


    return pose_list, cam_ext


def get_render_poses6():
    pose_list=[]   
        
    for deg in [-45]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (180-360,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 

    for deg in [45]:
        p=np.eye(4)
        p[:3,:3]=R.from_euler('yx', (240,-1*deg), degrees=True).as_matrix() 
        pose_list.append(p) 

    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (0, -100), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (0, -15), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    
    cam_ext=np.array([[ 1.,      0.,     0.,   -0.   ],
                      [-0.,     -1.,    -0.,    0.1  ],
                      [-0.,     -0.,    -1.,    2    ],
                      [ 0.,      0.,     0.,    1.0  ]])

    return pose_list, cam_ext


def get_render_poses7():
    pose_list=[]   
    
    
    # 45
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-130, 25), degrees=True).as_matrix() 
    pose_list.append(p) 

    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-220, 25), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    # front
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-90, 25), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    #left
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-175, 25), degrees=True).as_matrix() 
    pose_list.append(p)   


    return pose_list


def get_render_poses8():
    pose_list=[]   
    
    # 45
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-130, 25), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-90, 25), degrees=True).as_matrix() 
    pose_list.append(p) 
    
    p=np.eye(4)
    p[:3,:3]=R.from_euler('yx', (-30, 25), degrees=True).as_matrix() 
    pose_list.append(p)   

    return pose_list

def wtext(img,text):
    if img.dtype!=np.uint8:
        img=(img*255).astype(np.uint8)
        
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fsize=3
    thickn=6 
    top=80
    
    #cv2.putText(img, text, (100,100), font, fsize, (255,255,255),thickn+4)
    cv2.rectangle(img,     (100, 0), (100+len(text)*30, top), (255,255,255), -1)
    cv2.putText(img, text, (100, top), font, fsize, (0,0,255),thickn)
    
def add_hr(img):
    s=10
    img[:s, :,:]=0
    img[-s:,:,:]=0

def recenter_mesh(mesh): 
    mesh2=o3d.geometry.TriangleMesh(mesh)
    cen = -1*mesh.get_center()
    mesh2.translate(cen)
    return mesh2


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2





"""
Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT
"""

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length, resolution=4)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())

                #cylinder_segment = cylinder_segment.rotate(
                #    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)
    
    def get_mesh(self):
        m0 = self.cylinder_segments[0]
        
        for i in range(1,len(self.cylinder_segments)):
            m0+=self.cylinder_segments[i]
        return m0
        
    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def create_xz_plane(dim,y):
    gmin=-1.0
    gmax=1.0
    #dim=32
    # y=-0.8

    s=np.linspace(gmin,gmax,dim)
    x1=np.ones((dim,3))*y
    x2=np.ones((dim,3))*y
    x1[:,0]=s
    x2[:,0]=s
    x1[:,2]=gmin
    x2[:,2]=gmax
    z1=np.ones((dim,3))*y
    z2=np.ones((dim,3))*y
    z1[:,2]=s
    z2[:,2]=s
    z1[:,0]=gmin
    z2[:,0]=gmax

    pts=np.concatenate([x1,x2,z1,z2])
    lpts=np.zeros((dim*2,2),dtype=np.int32)
    lpts[:dim,0]=np.arange(dim)
    lpts[:dim,1]=np.arange(dim)+dim
    lpts[dim:,0]=np.arange(dim)+dim*2
    lpts[dim:,1]=np.arange(dim)+dim*3

    if 1:
        grid_plane=LineMesh(pts,lpts,radius=0.002).get_mesh() 
        grid_plane.paint_uniform_color((0.0,0.0,0.0))
    else:
        grid_plane = o3d.geometry.LineSet()
        grid_plane.points=o3d.utility.Vector3dVector( pts )
        grid_plane.lines=o3d.utility.Vector2iVector( lpts )
        grid_plane.paint_uniform_color((0.0,0.0,0.0))

    return grid_plane


def create_xy_plane(dim,z):
    gmin=-1.0
    gmax=1.0
    #dim=32
    # y=-0.8

    s=np.linspace(gmin,gmax,dim)
    x1=np.ones((dim,3))*z
    x2=np.ones((dim,3))*z
    x1[:,0]=s
    x2[:,0]=s
    x1[:,1]=gmin
    x2[:,1]=gmax

    y1=np.ones((dim,3))*z
    y2=np.ones((dim,3))*z
    y1[:,1]=s
    y2[:,1]=s
    y1[:,0]=gmin
    y2[:,0]=gmax

    # index
    pts=np.concatenate([x1,x2,y1,y2])
    lpts=np.zeros((dim*2,2),dtype=np.int32)
    lpts[:dim,0]=np.arange(dim)
    lpts[:dim,1]=np.arange(dim)+dim
    lpts[dim:,0]=np.arange(dim)+dim*2
    lpts[dim:,1]=np.arange(dim)+dim*3

    if 1:
        grid_plane=LineMesh(pts,lpts,radius=0.002).get_mesh() 
        grid_plane.paint_uniform_color((0.0,0.0,0.0))
    else:
        grid_plane = o3d.geometry.LineSet()
        grid_plane.points=o3d.utility.Vector3dVector( pts )
        grid_plane.lines=o3d.utility.Vector2iVector( lpts )
        grid_plane.paint_uniform_color((0.0,0.0,0.0))

    return grid_plane

def render_once(rdlist,x=0,y=0,z=0.2, pose_list=None, use_normal_shading=False, line_width=1, point_size=8, no_shading=False):
    vw=O3DVisWrap()
    
    vw.ropt.line_width=line_width
    vw.ropt.point_size=point_size
    vw.ropt.mesh_show_back_face=1

    if use_normal_shading:
        vw.ropt.mesh_color_option=vw.ropt.mesh_color_option.Normal

    if no_shading:
        vw.ropt.MeshShadeOption= vw.ropt.mesh_color_option.Default


    def_poses, cam_ext=get_render_poses6()
    cam_ext[0,3]=x
    cam_ext[1,3]=y
    cam_ext[2,3]=z
    
    if pose_list is None:
        pose_list = def_poses

    _img= vw.render_pose_list(rdlist,pose_list=pose_list, scale=1.0, ext=cam_ext)
    _img=(_img*255).astype(np.uint8) 
    
    return _img



def recenter(mesh): 
    mesh2=o3d.geometry.TriangleMesh(mesh)
    cen = -1*mesh.get_center()
    mesh2.translate(cen)
    return mesh2