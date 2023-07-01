import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
import sys
import os 
import math
from einops import rearrange, reduce, repeat
import numpy as np

#from .blocks import make_mlp, GeoSkipCMLP
#from .freq  import PosEncoder,  WindowedPosEncoder 

# https://github.com/cvg/nice-slam

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """
    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """
    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear',
                 color=False, color_out_dim=3,
                 skips=[2], 
                 #grid_len=0.16, 
                 pos_embedding_method='fourier', concat_feature=False):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        #self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 10
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, color_out_dim, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        # p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                          mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None):
        if self.c_dim != 0:
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()

        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_no_xyz(nn.Module):
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False,
                 sample_mode='bilinear', color=False, skips=[2]):
        super().__init__()
       
        # grid_len=0.16

        self.name = name
        self.no_grad_feature = False
        self.color = color
        #self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        # p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        if p.ndim==2:
            p_nor = rearrange(p,'n c -> 1 1 1 n c')
        elif p.ndim==3:
            p_nor = rearrange(p,'m n c -> 1 1 m n c')
        else:
            raise Exception('require update')

        vgrid = p_nor

        c = F.grid_sample(grid_feature, vgrid, 
                        padding_mode='border',
                        align_corners=True, mode=self.sample_mode)

        if p.ndim==3:
            c = rearrange(c,'1 c 1 m n -> m n c')
        elif p.ndim==2:
            c = rearrange(c,'1 c 1 1 n -> n c')
        else:
            raise Exception('require update')

        return c

    def forward(self, p, c_grid, **kwargs):

        # [,,,C]
        c = self.sample_grid_feature(p, c_grid['grid_' + self.name])
                
        h = c
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)

        out = self.output_linear(h)

        # if not self.color:
        #     out = out.squeeze(-1)

        return out



class NICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.

     coarse_grid_len=2.0,  
     middle_grid_len=0.16, fine_grid_len=0.16,
    """

    def __init__(self, dim=3, c_dim=32, color_grid_len=0.16, hidden_size=32, pos_embedding_method='fourier'):

        super().__init__()

        self.coarse=False
        if self.coarse:
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len)

        self.middle_decoder = MLP(name='middle', 
                                  dim=dim,
                                  c_dim=c_dim,
                                  color=False,
                                  skips=[2], 
                                  n_blocks=5, 
                                  hidden_size=hidden_size,
                                  #grid_len=middle_grid_len, 
                                  pos_embedding_method=pos_embedding_method)

        self.fine_decoder = MLP(name='fine', 
                                dim=dim,
                                c_dim=c_dim*2, 
                                color=False,
                                skips=[2], 
                                n_blocks=5, 
                                hidden_size=hidden_size,
                                #grid_len=fine_grid_len, 
                                concat_feature=True, 
                                pos_embedding_method=pos_embedding_method)

        self.color_decoder = MLP(name='color', 
                                 dim=dim, 
                                 c_dim=c_dim, 
                                 color=True,
                                 color_out_dim=3,
                                 skips=[2], 
                                 n_blocks=5, 
                                 hidden_size=hidden_size,
                                 #grid_len=color_grid_len, 
                                 pos_embedding_method=pos_embedding_method)

    def forward(self, p, c_grid, stage, **kwargs):
        """
            Output occupancy/color in different stage.
        """

        device = f'cuda:{p.get_device()}'

        if stage == 'coarse' and self.coarse:
            raw = self.coarse_decoder(p, c_grid)
            #occ = occ.squeeze(0)
            #raw = torch.zeros(occ.shape[0], 4).to(device).float()
            #raw[..., -1] = occ 

            rt={
                'raw':raw, 
            }
            return rt 

        elif stage == 'middle':
            middle_occ = self.middle_decoder(p, c_grid)  
            raw = middle_occ.unsqueeze(-1)

            rt={
                'raw':raw, 
            }
            return rt 
            
        elif stage == 'fine':
            
            fine_occ = self.fine_decoder(p, c_grid)
            
            middle_occ = self.middle_decoder(p, c_grid) 
            raw = fine_occ+middle_occ
            raw = raw.unsqueeze(-1)
            rt={
                'raw':raw, 
            }
            return rt 

        elif stage == 'color':
            
            rgb = self.color_decoder(p, c_grid)
            
            fine_occ   = self.fine_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid) 

            raw = (fine_occ+middle_occ)
            raw = raw.unsqueeze(-1)
            
            rt={
                'raw':raw,
                'rgb':rgb,
            }
            return rt 
        else:
            raise Exception('error mode:'+stage)


def make_grid( c_dim, reso, init_std ):
    grid = torch.zeros(1,c_dim, *reso)
    grid = grid.normal_(mean=0, std=init_std)
    grid = nn.Parameter(grid)
    return grid

# 
# model:
#   c_dim: 32
#   coarse_bound_enlarge: 2
#   pos_embedding_method: fourier
#   
class NICE_wraper(nn.Module): 

    def __init__(self, middle_reso, fine_reso, color_reso ):
        super().__init__() 

        assert isinstance(middle_reso,list)
        assert isinstance(fine_reso,list)
        assert isinstance(color_reso,list)

        c_dim=32
        self.decoder = NICE(c_dim=c_dim,  
                            pos_embedding_method='fourier')

        self.middle_grid= make_grid( c_dim, middle_reso,   init_std=0.01   )
        self.fine_grid  = make_grid( c_dim, fine_reso,  init_std=0.0001 )
        self.color_grid = make_grid( c_dim, color_reso, init_std=0.01   )

        # self.log_grad_plot_module=[]
        self.log_grad_module=[self.decoder.fine_decoder]  


    def forward(self,  tx_noc_p, stage ): 

        # def forward(self, p, c_grid, stage, **kwargs):
        # rt = self.NICE(p=tx_noc_p, 
        #                c_grid=,
        #                stage='fine')

        # win_a, 
        # t_code=None, 
        # amb_w=None, 
        # return_ft=False,

        # 'grid_coarse':
        c_grid={
            'grid_middle':self.middle_grid,
            'grid_fine':self.fine_grid,
            'grid_color':self.color_grid,
        }

        rt = self.decoder(p=tx_noc_p, 
                       c_grid=c_grid,
                       stage=stage)  

        return rt 

class IMap_wraper(nn.Module): 

    def __init__(self, ):
        super().__init__() 

        # decoder = models.decoder_dict['imap'](
        #             dim=dim, c_dim=0, color=True,
        #             hidden_size=256, skips=[], n_blocks=4, pos_embedding_method=pos_embedding_method
        #         )

        self.decoder = MLP( dim=3, 
                            c_dim=0, 
                            color=True, 
                            color_out_dim=4,
                            hidden_size=256, 
                            skips=[], 
                            n_blocks=4, 
                            pos_embedding_method='fourier') 

        # self.log_grad_plot_module=[]
        self.log_grad_module=[self.decoder.pts_linears]  


    def forward(self,  tx_noc_p, stage=None ): 

        y = self.decoder(p=tx_noc_p )  

        rt ={}
        rt['rgb'] = y[...,:3]
        rt['raw'] = y[...,-1:]
        rt['sigma'] = rt['raw'] 

        return rt 


#-----------------------------------------------------------
class MainMLP(nn.Module):
    #
    #  proxy class
    # 
    def __init__(self, use_stage ):

        super().__init__() 
        self.use_bending= False 
        self.append_t =False

        self._deform_win_a=1.0
        self._deform_win_b=1.0

        self.use_stage=use_stage
        self.stage='color'

        # self.skip_outter_samples = skip_outter_samples
        self.log_grad_module=[]  

    def set_stage(self, st):
        assert st in ['middle','fine','color']
        self.stage=st

    def convert_coord(self, p, min_bound, grid_size):  
        assert p.ndim==2
        assert min_bound.ndim==2
        assert grid_size.ndim==2
        
        max_v = grid_size.max()
        # noc_p = ((p - min_bound)/grid_size)*2-1.0
        noc_p = ((p - min_bound)/max_v)*2-1.0
        return noc_p 
        
    def forward(self, p, vdirs, scene_data, normalize, motion_net, p_tidx, deform=False, sigma_only=False ):
        
        assert p.ndim==2 
        if vdirs is not None:
            assert p.shape==vdirs.shape, f'{p.shape} {vdirs.shape}'

        all_ft=[]
        all_valid=[]

        sdf_decoder = scene_data['model']
        grid_size = scene_data['grid_size']
        min_bound = scene_data['min_bound']
        
        # grid_size = sdf_decoder.grid_size
        # min_bound = sdf_decoder.min_bound

        #--------------------------------------------
        # normalize points
        if normalize:
            noc_p = self.convert_coord(p, min_bound, grid_size)
        else:
            noc_p = p 

        #--------------------------------------------
        # deformation block

        if deform:
            assert motion_net is not None
            assert p_tidx is not None
            assert p_tidx.shape[0] == noc_p.shape[0]

            if self.use_bending:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a,
                                    win_b=self._deform_win_b,)
                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = rt['w']

            else:
                rt = motion_net.apply_deform_batch(
                                    t_idx=p_tidx, 
                                    p=noc_p, 
                                    win_a=self._deform_win_a )

                tx_noc_p= rt['q']
                p_shift = rt['delta']
                amb_w   = None

        else:
            p_shift = None 
            tx_noc_p= noc_p

            if self.use_bending:

                amb_w = repeat(self.rigid_amb_w, 
                              'c -> n c', n=noc_p.shape[0]).clone()

            else:
                amb_w = None

        #--------------------
        # n,c 
        if self.append_t :
            # n,t
            t_code = motion_net.get_time_code(p_tidx)
        else:
            t_code = None

        #--------------------------------------------        
        #if tx_noc_p.requires_grad ==False:
        #    tx_noc_p.requires_grad=True
        
        # in-bound check  
        _valid =(-1<=tx_noc_p)*(tx_noc_p<=1)
        _valid =reduce(_valid,'n c -> n', 'sum')
        va = _valid==3

        #--------------------------------------------
        if self.use_stage and (self.stage=='middle' or self.stage=='fine'):
            sout  = sdf_decoder(tx_noc_p=tx_noc_p, stage=self.stage)

            p_raw = sout['raw']
            p_rgb = torch.zeros((p_raw.shape[0],3),device=p_raw.device)

        else:
            sout  = sdf_decoder(tx_noc_p=tx_noc_p, stage=self.stage) 

            p_raw = sout['raw']
            p_rgb = sout['rgb'] 


        out={ 
            'sigma':p_raw,
            'raw':p_raw, 
            'rgb':p_rgb,
            'normal':None, 
            'p_shift':p_shift,
            '_valid':va,
            '_noc_p':noc_p,
            '_tx_noc_p':tx_noc_p,
        }
        return out 
