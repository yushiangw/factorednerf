import numpy as np

#import neptune
import time
import pdb
import torch
import torch.utils
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import os

class TBWriters:

    def __init__(self, exp_name, log_dir, only_train=False):
        self.exp_name=exp_name
        
        if self.exp_name is None:
            self.exp_name=''

        tr= os.path.join(log_dir,'train')
        val= os.path.join(log_dir,'val')
        os.makedirs(tr,exist_ok=True)

        self.train_writer = SummaryWriter(log_dir=tr, filename_suffix=self.exp_name , flush_secs=10, )
        self.only_train=only_train

        if not only_train:
            os.makedirs(val,exist_ok=True)
            self.val_writer   = SummaryWriter(log_dir=val, filename_suffix=self.exp_name , flush_secs=10 )
        
        self._last = time.time() 
        self._cache=[]

    def __w_scalar(self, mode, group_name, name, value,  step): 
        cname = group_name +'/' +name

        if self.only_train:
            self.train_writer.add_scalar(cname,value,step)

        else:
            if mode =='train' :
                self.train_writer.add_scalar(cname,value,step)
            elif mode=='val':
                self.val_writer.add_scalar(cname,value,step)

    def add_scalar(self, mode, group_name, name, value,  step): 
        # mode=none -> single writer mode 

        if time.time() - self._last < 10:
            # cache 
            self._cache.append((mode, group_name, name, value,  step)) 
            return 
            
        else:
            # first write cache 
            for x in self._cache:
                m,g,n,v,s  = x
                self.__w_scalar( m,g,n,v,s ) 
            self._cache=[]

            self.__w_scalar(mode, group_name, name, value, step)
            self._last=time.time()

        #-----------------------------

    def add_image(self, mode, group_name, name, value, step, dataformats='CHW'):

        cname = name

        if self.only_train:
            self.train_writer.add_image( 
                cname,value,step, dataformats=dataformats) 
        else:
            if mode =='train':
                self.train_writer.add_image( 
                    cname,value,step, dataformats=dataformats)
            elif mode=='val':
                self.val_writer.add_image( 
                    cname,value,step, dataformats=dataformats)

    def add_figure(self, mode, group_name, name, value,  step ):

        cname = name
        
        if self.only_train:
            self.train_writer.add_figure( cname,value,step )
        else:
            if mode =='train':
                self.train_writer.add_figure( cname,value,step )
            elif mode=='val':
                self.val_writer.add_figure( cname,value,step )



class TBOverlayWriters:

    def __init__(self, exp_name, log_dir, only_train=False):
        self.exp_name=exp_name

        if self.exp_name is None:
            self.exp_name=''

        self.tr_dir = os.path.join(log_dir,'train')
        self.val_dir= os.path.join(log_dir,'val')

        os.makedirs(self.tr_dir,exist_ok=True)

        # self.train_writer = SummaryWriter(log_dir=tr, filename_suffix=self.exp_name , flush_secs=10, )
        self.only_train=only_train
        if not only_train:
            os.makedirs(self.val_dir,exist_ok=True) 
            #self.val_writer   = SummaryWriter(log_dir=val, filename_suffix=self.exp_name , flush_secs=10 )
        
        self.writer_dict={}

        self._last = time.time() 
        self._cache=[]

    def get_wt(self, mode, name ):
        
        #x = name.split('/')
        #cname = x[0] #'_'.join(x[:1])

        tag = f'{mode}/{name}'

        if mode=='train':
            log_dir=self.tr_dir
        else:
            log_dir=self.val_dir

        if tag not in self.writer_dict:

            vname = name.replace('/','_')
            vname = vname.replace('\\','_')
            
            self.writer_dict[tag] = SummaryWriter(
                                        log_dir=os.path.join(log_dir,vname),
                                        flush_secs=10 )
        

        return self.writer_dict[tag] 


    def __w_scalar(self, mode, group_name, name, value,  step): 
        wt = self.get_wt(mode, group_name)

        cname = name

        if self.only_train:
            wt.add_scalar(cname,value,step)

        else:
            if mode =='train' :
                wt.add_scalar(cname,value,step)
            elif mode=='val':
                wt.add_scalar(cname,value,step)

    def add_scalar(self, mode, group_name, name, value,  step): 
        # mode=none -> single writer mode  


        if time.time() - self._last < 10:
            # cache 
            self._cache.append((mode, group_name, name, value,  step)) 
            return 
            
        else:
            # first write cache 
            for x in self._cache:
                m,gn,n,v,s  = x
                self.__w_scalar( m,gn,n,v,s ) 

            self._cache=[]

            self.__w_scalar(mode, group_name, name, value, step)
            self._last=time.time()

    def add_image(self, mode, group_name, name, value,  step, dataformats='CHW'):
        wt = self.get_wt(mode, group_name)

        cname = name
            

        if self.only_train:
            wt.add_image( cname,value,step, dataformats=dataformats)

        else:
            if mode =='train':
                wt.add_image( cname,value,step, dataformats=dataformats)
            elif mode=='val':
                wt.add_image( cname,value,step, dataformats=dataformats)

    def add_figure(self, mode, group_name, name, value,  step ):
        wt= self.get_wt(mode, group_name)

        cname = name 

        if self.only_train:
            wt.add_figure( cname,value,step )
        else:
            if mode =='train':
                wt.add_figure( cname,value,step )
            elif mode=='val':
                wt.add_figure( cname,value,step )


class NeptWriters:

    def __init__(self, user_proj,  exp_name, description, tags=None):
        self.exp_name=exp_name

        #torch.utils.tensorboard.writer.
        #SummaryWriter(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')

        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjAyYzgzNjAtODBhNS00MjJhLWEyOGUtYmRiNGI1ZTcxNTc0In0='
        
        # 'ysw/sandbox'
        neptune.init(user_proj, api_token=api_token)
        self.exp=neptune.create_experiment(name=exp_name,tags=tags,description=description)

        self._step_log={}

    def add_scalar(self, mode, name, value,  step): 
        '''
            step are synchronized 
        '''
        if mode is not None:
            tag_name = '{:}/{:}'.format(mode, name )
            step_name = '{:}/step'.format(mode )
        else:
            tag_name=name
            step_name='step'

        neptune.log_metric(tag_name, value )

        if step not in self._step_log:
            neptune.log_metric(step_name, step  )
            self._step_log[step]=True

    def add_np_image(self, mode, name, value,  step, dataformats):

        # fig
        neptune.log_image('{:}/{:}'.format(mode, name), value)

    def __del__(self):  
        self.exp.stop()

        # if mode =='train':
        #     self.train_writer.add_image( name,value,step, dataformats=dataformats)
        # elif mode=='val':
        #     self.val_writer.add_image( name,value,step, dataformats=dataformats)
