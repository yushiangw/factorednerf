#!/usr/bin/python3 
import os,sys,json,time,types,math,gc,pdb,glob 
import numpy as np 
import pyvista as pv

v1_root='/ysmedia/sync/dataset/ShapeNetCore.v1/'


def print_cam(plotter, name):
    
    print(f'{name}.camera.position=',plotter.camera.position)
    print(f'{name}.camera.focal_point=',plotter.camera.focal_point)
    print(f'{name}.camera.up=',plotter.camera.up)
    
def set_cam(plotter, zidx):
    # 2.9
    #plotter.camera.position=(-2.9624187279390837, 2.8763158120179413, zidx)
    #plotter.camera.focal_point=(0.0, 0.0, 0.0)
    #plotter.camera.up=(0.10340380211431388, 0.7661964999674937, -0.6342322738128878)
    #plotter.camera.position= (-0.7575313581305706, 1.1609670200227051, 3.272096844758223)
    #plotter.camera.focal_point= (0.0, 0.0, 0.0)
    #plotter.camera.up= (0.09829938967217443, 0.9448263418309848, -0.3124746610084681)

    plotter.camera.position= (-2.142179858331498, 2.3769910913053467, 1.5457343717541279)
    plotter.camera.focal_point= (0.0, 0.0, 0.0)
    plotter.camera.up= (0.3779656481929343, 0.7195050944523496, -0.582627143071116)


def add_mesh(plotter,fp,color,wireframe=False,use_vertex_color=False):
    
    m1=pv.read(fp)  
    print(fp)
    
    if use_vertex_color:
        if wireframe:
            plotter.add_mesh(m1, rgb=True, style='wireframe')
        else:
            plotter.add_mesh(m1, rgb=True )

    else:
        if wireframe:
            plotter.add_mesh(m1, color=color, style='wireframe')
        else:
            plotter.add_mesh(m1, color=color, )
        
    # plotter.camera.position = (0, 1.0, zz)
    # plotter.camera.focal_point = (0.0, 0.0, 0.0)
    # plotter.camera.up = (0.0, 1.0, 0.0) 

    
def add_pts(plotter,pts,color=None,rgb=None,point_size=10, nv=None, nv_size=0.15):
    
    pcd = pv.PolyData(pts)  
    if rgb is not None:
        pcd['rgb']=rgb
        plotter.add_mesh(pcd, rgb=True, render_points_as_spheres=True,point_size=point_size)  
    elif color is not None:
        plotter.add_mesh(pcd, color=color, render_points_as_spheres=True,point_size=point_size)  
    else:
        plotter.add_mesh(pcd, render_points_as_spheres=True,point_size=point_size)  

    if nv is not None:
        pcd['vectors'] = nv
        arrows = pcd.glyph(
                    orient='vectors',
                    scale=False,
                    factor=nv_size,
                )
        plotter.add_mesh(arrows, color='lightblue')
        



    # plotter.camera.position = (0, 1.0, zz)
    # plotter.camera.focal_point = (0.0, 0.0, 0.0)
    # plotter.camera.up = (0.0, 1.0, 0.0) 

def add_rays(plotter, rays_o, rays_d, z ,width, color ):

    for k in range(rays_d.shape[0]):
        p1 = rays_o[k] 
        p2 = rays_o[k]+ rays_d[k]*z 
        points = np.array([p1,p2])
        plotter.add_lines(points, color=color, width=width)


def add_rays_ab(plotter, rays_o, rays_d, za, zb, width, color ):

    for k in range(rays_d.shape[0]):
        p1 = rays_o[k]+ rays_d[k]*za
        p2 = rays_o[k]+ rays_d[k]*zb 
        points = np.array([p1,p2])
        plotter.add_lines(points, color=color, width=width)


def add_lines(plotter, pcd1, pcd2, color, width=1, ):

    for k in range(pcd1.shape[0]):
        p1 = pcd1[k] 
        p2 = pcd2[k]
        points = np.array([p1,p2])
        plotter.add_lines(points, color=color, width=width)


def add_obb(plotter, bmin, bmax, color=None, width=1):

    ax,ay,az=bmin
    bx,by,bz=bmax

    obb_lines=[
        ([ax,ay,az],[bx,ay,az]), # bottom
        ([ax,ay,az],[ax,by,az]), # bottom
        ([bx,ay,az],[bx,by,az]), # bottom
        ([ax,by,az],[bx,by,az]), # bottom
        ([ax,ay,bz],[bx,ay,bz]), # top
        ([ax,ay,bz],[ax,by,bz]), # top
        ([bx,ay,bz],[bx,by,bz]), # top
        ([ax,by,bz],[bx,by,bz]), # top
        ([ax,ay,az],[ax,ay,bz]), # top
        ([ax,by,az],[ax,by,bz]), # top
        ([bx,ay,az],[bx,ay,bz]), # top
        ([bx,by,az],[bx,by,bz]), # top
    ]


    for k in range(len(obb_lines)):
        points = obb_lines[k]
        points = np.array(points)

        plotter.add_lines(points, color=color, width=width)


def add_aabb(plotter, bmin, bmax, color=None):

    bounds=[]
    for i in range(3):
        bounds.append(bmin[i])
        bounds.append(bmax[i])

    mesh = pv.Box( bounds=bounds, level=0)
    plotter.add_mesh(mesh, color=color, style='wireframe')


def display_mesh_pcd( mfp_list, pcd_list=None, nv_list=None, deg=50, live=False, zz=2):    

    ws = 600
    
    pv.set_plot_theme("document") 
    num = len(mfp_list)
    plotter = pv.Plotter(shape=(2,num), 
                    window_size=(ws*num,ws), notebook=True)
    
    k=0
    for f in mfp_list: 
        plotter.subplot(0, k)
        
        mesh_1 = pv.read(f)  
        mesh_1.rotate_y(deg)  
        plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
        
        plotter.camera.position = (0, 1.0, -3.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)

        #----------------------------------

        plotter.subplot(1, k)
        mesh_1 = pv.read(f)  
        mesh_1.rotate_y(deg)  
        plotter.add_mesh(mesh_1, opacity=0.5, color=(0.7,0.7,1.0))
        
        plotter.camera.position = (0, 1.0, zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)

        if pcd_list is not None:
            pts=pcd_list[k]
            nv =nv_list[k]

            if pts is not None:
                pcd = pv.PolyData(pts) 
                if nv is not None:
                    pcd['nv']=nv
                    pcd.set_active_vectors("nv")
                pcd.rotate_y(deg)  
                plotter.add_mesh(pcd, color=(1.0,0.7,0.7), render_points_as_spheres=True)  
            
                if nv is not None:
                    arrows = pcd.glyph(orient='nv', scale=False, factor=0.1,) 
                    plotter.add_mesh(arrows, lighting=False )       

        plotter.camera.position = (0, 1.0, zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        k+=1
    
    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()


def display_v1_samples(cl, mid, pts1, nv1, pts2, nv2, tx=0, scale=1.0, live=False):
    mfp = os.path.join(root, cl, mid+'/model.obj' )
    ws = 600 
    pv.set_plot_theme("document") 
    num = 1
    plotter = pv.Plotter(shape=(1,num), window_size=(ws*num,ws), notebook=True)
    
    deg=60
    plotter.subplot(0,0)
    
    k=0
    for pts,nv in [ (pts1,nv1), (pts2,nv2)]:
        
        pcd = pv.PolyData(pts)
        if nv is not None:
            pcd['nv']=nv
            pcd.set_active_vectors("nv")
        pcd.rotate_y(deg) 
        #======================================================
        if k==0:
            color=(0,1.0,0)
        else:
            color=(1.0,0.0,0.0)
            
        plotter.add_mesh(pcd, color=color, render_points_as_spheres=True)  

        if nv is not None:
            arrows = pcd.glyph(orient='nv', scale=False, factor=0.2,) 
            plotter.add_mesh(arrows, lighting=False )       
           
        #======================================================        
        k+=1
        
    mesh_1 = pv.read(mfp)  
    mesh_1.translate(tx)
    mesh_1.scale(scale)
    mesh_1.rotate_y(deg)   
    plotter.add_mesh(mesh_1, opacity=0.5, color=(0.9,0.9,0.9))

    plotter.camera.position = (0, 1.0, -3.0)
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.show()
    if not live:
        plotter.close()

        
        
def display_v1_wpcd(cl, mid, pts, nv=None, tx=0, scale=1.0, live=False):
    mfp = os.path.join(root, cl, mid+'/model.obj' )
    ws = 600
    
    pv.set_plot_theme("document") 
    num = 1
    plotter = pv.Plotter(shape=(1,num), window_size=(ws*num,ws), notebook=True)
    
    deg=60
    k=0
    for pc in [mfp]: 
        
        plotter.subplot(0, k)
        
        pcd = pv.PolyData(pts)
        if nv is not None:
            pcd['nv']=nv
            pcd.set_active_vectors("nv")
        pcd.rotate_y(deg)  
        plotter.add_mesh(pcd, color=(0.7,0.7,1.0), render_points_as_spheres=True)  
        
        if nv is not None:
            arrows = pcd.glyph(orient='nv', scale=False, factor=0.1,) 
            plotter.add_mesh(arrows, lighting=False )       
            
        mesh_1 = pv.read(pc)  
        mesh_1.translate(tx)
        mesh_1.scale(scale)
        mesh_1.rotate_y(deg)  
        
        plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
        
        plotter.camera.position = (0, 1.0, -3.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        k+=1
    
    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()
    
    
    
def display_v1(cl,mid):    
    mfp = os.path.join(root, cl, mid+'/model.obj' )
    ws = 600
    
    pv.set_plot_theme("document") 
    num = 1
    plotter = pv.Plotter(shape=(1,num), window_size=(ws*num,ws), notebook=True)
    
    deg=60
    k=0
    for pc in [mfp]: 
        mesh_1 = pv.read(pc)  
        mesh_1.rotate_y(deg)  
        plotter.subplot(0, k)
        
        if k==1:
            plotter.add_mesh(mesh_1, color=(1.0,0.7,0.0))
        else:
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
        
        plotter.camera.position = (0, 1.0, -3.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        k+=1
    
    #------------------------------- 
    plotter.show()
    plotter.close()
    

def display_v1_wmfp(cl, mid, mfp_list, tx=0, scale=1.0, live=False):    
    _mfp = os.path.join(root, cl, mid+'/model.obj' )
    ws = 600
    
    pv.set_plot_theme("document") 
    num = len(mfp_list)+1
    plotter = pv.Plotter(shape=(1,num), window_size=(ws*num,ws), notebook=True)
    
    deg=60
    k=0
    for pc in [_mfp]+mfp_list: 
        plotter.subplot(0, k)
        
        mesh_1 = pv.read(pc)  
        
        if k==0:
            mesh_1.translate(tx)
            mesh_1.scale(scale)
            plotter.add_mesh(mesh_1, color=(1.0,0.7,0.0))
        else:
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
        
        mesh_1.rotate_y(deg)  
        plotter.camera.position = (0, 1.0, -3.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        k+=1
    
    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()
        
        
def display_mesh_pcdv3(mfp_list,
                       pcd_list=None, 
                       nv_list=None, degs=[50], zz=-1.5, 
                       live=False, point_size=10, wm=False, alpha=0.3):    

    ws=600 
    pv.set_plot_theme("document") 
    num = len(mfp_list)*len(degs)
    plotter = pv.Plotter(shape=(2,num), window_size=(ws*num,ws), notebook=True)
    
    k=-1
    for dd in degs:
        for i,f in enumerate(mfp_list): 
            k+=1
            plotter.subplot(0, k)

            mesh_1 = pv.read(f)  
            mesh_1.rotate_y(dd)  
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))

            plotter.camera.position = (0, 1.0, zz)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.camera.up = (0.0, 1.0, 0.0)
            
            #=========================
            plotter.subplot(1, k) 
            
            if pcd_list is not None:
                pts=pcd_list[i]

                if pts is not None:
                    pcd = pv.PolyData(pts)

                if nv_list is not None and nv_list[i] is not None:
                    nv =nv_list[i]
                    pcd['nv']=nv
                    pcd.set_active_vectors("nv")
                
                pcd.rotate_y(dd)  
                plotter.add_mesh(pcd, color=(1.0,0.7,0.7), render_points_as_spheres=True, point_size=point_size)  
                
                #=========================
                if nv_list is not None and nv_list[i] is not None:
                    arrows = pcd.glyph(orient='nv', scale=False, factor=0.1,) 
                    plotter.add_mesh(arrows, lighting=False )       

                if wm:  
                    plotter.add_mesh(mesh_1, opacity=alpha, color=(0.7,0.7,1.0))

            plotter.camera.position = (0, 1.0,zz)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.camera.up = (0.0, 1.0, 0.0) 

    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()



def display_mesh_pcdv4(mfp_list,
                       pcd_list=None, 
                       nv_list=None, 
                       degs=[50], 
                       zz=-1.5, 
                       live=False,
                       point_size=10,
                       wm=False,
                       alpha=0.3,
                       save=False   ):    

    ws=600 
    pv.set_plot_theme("document") 
    num = len(degs)
    plotter = pv.Plotter(shape=(2,num), window_size=(ws*num,ws), notebook=True)
    
    k=-1
    for dd in degs:
        k+=1
        
        plotter.subplot(0, k)
        
        for f in mfp_list: 
            mesh_1 = pv.read(f)  
            mesh_1.rotate_y(dd)  
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
        
        plotter.camera.position = (0, 1.0, zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        
        #===================================================
        plotter.subplot(1, k)
        for f in mfp_list: 
            mesh_1 = pv.read(f)  
            mesh_1.rotate_y(dd)  
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0))
            
            if save :
                mesh_1.save('temp/mesh_1.ply')  
                
        for ix,pts in enumerate(pcd_list):
            
            if pts is not None:
                pcd = pv.PolyData(pts)

            if nv_list is not None and nv_list[ix] is not None:
                nv =nv_list[ix]
                pcd['nv']=nv
                pcd.set_active_vectors("nv")
            else:
                nv = None 
                
            pcd.rotate_y(dd)  
            plotter.add_mesh(pcd, color=(1.0,0.7,0.7), render_points_as_spheres=True, point_size=point_size)  
            
            if save:
                pcd.save(f'temp/pcd_{ix:d}.ply')  

            if nv is not None:
                arrows = pcd.glyph(orient='nv', scale=False, factor=0.1,) 
                plotter.add_mesh(arrows, lighting=False, show_scalar_bar=False)      
                if save: 
                    pcd.save(f'temp/arrows_{ix:d}.ply')  
                
            if wm:  
                plotter.add_mesh(mesh_1, opacity=alpha, color=(0.7,0.7,1.0))

        plotter.camera.position = (0, 1.0,zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0) 

    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()
        

def display_mesh_pcdv5(mfp_list,
                       pcd_list=None, 
                       nv_list=None, 
                       degs=[50], 
                       zz=-1.5, 
                       live=False,
                       point_size=10,
                       wm=False,
                       alpha=0.3,
                       rgb=False,
                       save=False,
                       name=None ):    

    ws=600 
    pv.set_plot_theme("document") 
    num = len(degs)
    
    rnum =2 if pcd_list is not None else 1
    
    plotter = pv.Plotter(shape=(rnum,num), window_size=(ws*num,ws), notebook=True)
    
    k=-1
    for dd in degs:
        k+=1
        
        plotter.subplot(0, k)
        
        for f in mfp_list: 
            mesh_1 = pv.read(f)  
            mesh_1.rotate_y(dd)  
            if rgb:
                plotter.add_mesh(mesh_1, rgb=rgb)
            else:
                plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0), rgb=rgb)
        
        plotter.camera.position = (0, 1.0, zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
           
        if k==0 and name is not None:
            actor = plotter.add_text( name , position='upper_left', color='blue',
                    shadow=True, font_size=14)
        
        #===================================================
        if pcd_list is None:
            continue 
            
        plotter.subplot(1, k)
        for f in mfp_list: 
            mesh_1 = pv.read(f)  
            mesh_1.rotate_y(dd)  
            plotter.add_mesh(mesh_1, color=(0.7,0.7,1.0), rgb=rgb)
            
            if save :
                mesh_1.save('temp/mesh_1.ply')  
                
        for ix,pts in enumerate(pcd_list):
            
            if pts is not None:
                pcd = pv.PolyData(pts)

            if nv_list is not None and nv_list[ix] is not None:
                nv =nv_list[ix]
                pcd['nv']=nv
                pcd.set_active_vectors("nv")
            else:
                nv = None 
                
            pcd.rotate_y(dd)  
            plotter.add_mesh(pcd, color=(1.0,0.7,0.7), render_points_as_spheres=True, point_size=point_size)  
            
            if save:
                pcd.save(f'temp/pcd_{ix:d}.ply')  

            if nv is not None:
                arrows = pcd.glyph(orient='nv', scale=False, factor=0.1,) 
                plotter.add_mesh(arrows, lighting=False, show_scalar_bar=False)      
                if save: 
                    pcd.save(f'temp/arrows_{ix:d}.ply')  
                
            if wm:  
                plotter.add_mesh(mesh_1, opacity=alpha, color=(0.7,0.7,1.0))

        plotter.camera.position = (0, 1.0,zz)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0) 

    #------------------------------- 
    plotter.show()
    if not live:
        plotter.close()