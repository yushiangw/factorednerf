import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange, reduce, repeat 

def poc_encoding(x, poc_scale):
    # https://github.com/hustvl/TiNeuVox/blob/381f931df135b02f6c92e94594defbdc1f9b0afc/lib/tineuvox.py#L66
    
    assert x.ndim==2
    assert poc_scale.ndim==1

    shape =x.shape

    K = poc_scale.shape[0]

    # x : N C 
    # poc_scale: K

    x2  = rearrange(x, 'n c -> n c 1')
    ps2 = rearrange(poc_scale,'k -> 1 1 k')

    # n,c,k
    emb = (x2* ps2)

    x_sin = emb.sin()
    x_cos = emb.cos()

    # n,c,k*2
    enc = torch.cat([x_sin,x_cos], -1)
    enc = rearrange(enc, ' n c k -> n (c k)')

    # n,c+c*k*2
    enc = torch.cat([x, enc],dim=-1)

    return enc


class PosEncoder(nn.Module):

    def __init__(self, in_dim, pe_dim ):
        super().__init__() 

        _scale2 = [ 2**(i+1) for i in range(pe_dim)]
        _scale2 = torch.tensor(_scale2,dtype=torch.float32)

        self.register_buffer('enc_scale', _scale2)

        self.out_dim = in_dim + in_dim*pe_dim*2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, p):

        # [p, ft]
        p2 = poc_encoding(p, self.enc_scale)

        return tz   
        

# https://github.com/google/hypernerf/blob/d433ebeba4ddd91fd83aa9af3423333d2d5934e7/hypernerf/model_utils.py#L342 

def posenc_window(min_deg, max_deg, alpha):
  """Windows a posenc using a cosiney window.
  This is equivalent to taking a truncated Hann window and sliding it to the
  right along the frequency spectrum.
  Args:
    min_deg: the lower frequency band.
    max_deg: the upper frequency band.
    alpha: will ease in each frequency 
            as alpha goes from min_deg to max_deg.
  Returns:
    (N-1)
    A 1-d numpy array with num_sample elements containing the window.
  """

  bands = torch.arange(min_deg, max_deg)
  
  x = torch.clip(alpha - bands, 0.0, 1.0)

  #  smoothly annealing depends on x [0.2->0.1]
  # 
  return 0.5 * (1 + torch.cos( np.pi * x + np.pi))

    
def sin_pos_enc(x, min_deg, max_deg, use_identity, alpha):

    # F,
    scales = 2.0 ** torch.arange(min_deg, max_deg).to(x.device)

    # (*, F, C).
    xb = x[..., None, :] * scales[:, None]

    # (*, F, 2, C).
    ff_x = torch.stack([xb, xb + 0.5 * np.pi], dim=-2)
    four_feat = torch.sin(ff_x)

    if alpha is not None:
        # alpha in [min,max]
        # (F,)
        window = posenc_window(min_deg, max_deg, alpha).to(x.device)
        four_feat = window[..., None, None] * four_feat

    # (*, F, 2, C). to (*,Fx2xC)
    four_feat = rearrange(four_feat,'... f k c -> ... (f k c)')

    if use_identity:
        return torch.cat([x, four_feat], dim=-1)
    else:
        return four_feat


class WindowedPosEncoder(nn.Module):

    def __init__(self, in_dim, min_deg, max_deg, use_identity):
        super().__init__() 
        
        self.min_deg=min_deg
        self.max_deg=max_deg
        self.use_identity=use_identity

        num = max_deg-min_deg
        assert num>0

        if use_identity:
            self.out_dim = in_dim + in_dim*num*2
        else:
            self.out_dim = in_dim*num*2


    def get_output_dim(self):
        return self.out_dim

    def forward(self, p, alpha=None):

        x=sin_pos_enc(p, self.min_deg, self.max_deg,
                      use_identity=self.use_identity, alpha=alpha)

        return x 