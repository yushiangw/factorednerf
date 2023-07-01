import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
from mpl_toolkits.axes_grid1 import make_axes_locatable


class VisualizerEval(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,  decoders, use_depth_sample=True):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        #with torch.no_grad():
    
        #if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
        if 1:
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
        
            if len(c2w_or_camera_tensor.shape) == 1:
                bottom = torch.from_numpy(
                    np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                        torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    c2w_or_camera_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            else:
                c2w = c2w_or_camera_tensor


            if use_depth_sample:
                depth, uncertainty, color = self.renderer.render_img(
                                                c,
                                                decoders,
                                                c2w,
                                                self.device,
                                                stage='color',
                                                gt_depth=gt_depth)
            else:
                depth, uncertainty, color = self.renderer.render_img(
                                                c,
                                                decoders,
                                                c2w,
                                                self.device,
                                                stage='color',
                                                gt_depth=None)


            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()
            
            inv_mask = gt_depth_np == 0.0

            #=====================================================================
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[inv_mask] = 0.0
            
            val_num = np.sum(~inv_mask)

            depth_err = depth_residual[~inv_mask].mean()

            val_dep_rd = depth_residual[~inv_mask]

            d_acc1 = val_dep_rd<5e-2
            d_acc1 = d_acc1.sum()/val_num

            d_acc2 = val_dep_rd<1e-2
            d_acc2 = d_acc2.sum()/val_num
            
            d_acc3 = val_dep_rd<1e-3
            d_acc3 = d_acc3.sum()/val_num
             
            #=====================================================================
            color_residual = np.abs(gt_color_np - color_np)
            color_residual[inv_mask] = 0.0
            color_residual = color_residual.mean(axis=-1)  
            color_err = color_residual.mean() 

            val_color_rd = color_residual[~inv_mask]

            color_acc1 = val_color_rd<0.05
            color_acc1 = color_acc1.sum()/val_num

            color_acc2 = val_color_rd<0.01
            color_acc2 = color_acc2.sum()/val_num

            color_acc3 = val_color_rd<0.001
            color_acc3 = color_acc3.sum()/val_num
            #=====================================================================
            
            txt_out = f'{self.vis_dir}/eval_{idx:05d}_{iter:04d}.txt'

            with open(txt_out,'w') as fout:
                ll=''
                ll+='depth_err,color_err,d_acc1,d_acc2,d_acc3,color_acc1,color_acc2,color_acc3\n'.replace(',','\t')
                for v in [depth_err,color_err]:
                    ll += f'{v:.2e}\t'
                for v in [d_acc1,d_acc2,d_acc3,color_acc1,color_acc2,color_acc3]:
                    ll += f'{v:.2e}\t'
                ll+= '\n\n'
                ll+= f'depth error:  {depth_err:.2e}\n'
                ll+= f'depth acc(5e-2): {d_acc1:.2f}\n'
                ll+= f'depth acc(1e-2): {d_acc2:.2f}\n'
                ll+= f'depth acc(1e-3): {d_acc3:.2f}\n'

                ll+= f'color error: {color_err:.2e}\n'
                ll+= f'color acc(0.05): {color_acc1:.2f}\n'
                ll+= f'color acc(0.01): {color_acc2:.2f}\n'
                ll+= f'color acc(0.001): {color_acc3:.2f}\n'
                fout.write(ll)

            #=====================================================================
            plt.figure(figsize=(18,12), dpi=600)

            fig, axs = plt.subplots(2, 3)
            fig.tight_layout()
            # plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace = 0.1)

            max_depth = np.max(gt_depth_np)
            axs[0, 0].imshow(gt_depth_np, cmap="turbo",
                             vmin=0, vmax=max_depth)
            axs[0, 0].set_title('Input Depth')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].imshow(depth_np, cmap="turbo", vmin=0, vmax=max_depth)
            axs[0, 1].set_title('Generated Depth')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            
            _im=axs[0, 2].imshow(depth_residual, cmap="viridis", vmin=0, vmax=0.05)

            divider = make_axes_locatable(axs[0, 2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(_im, cax=cax, orientation='vertical')

            axs[0, 2].set_title('Depth Residual')
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            axs[0, 2].set_xlabel(f'e={depth_err:.2e}')
            gt_color_np = np.clip(gt_color_np, 0, 1)
            
            color_np = np.clip(color_np, 0, 1)
            #color_residual = np.clip(color_residual, 0, 1)

            axs[1, 0].imshow(gt_color_np)
            axs[1, 0].set_title('Input RGB')
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
            axs[1, 1].imshow(color_np)
            axs[1, 1].set_title('Generated RGB')
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])
            
            _im=axs[1, 2].imshow(color_residual, cmap="plasma", vmin=0, vmax=0.2)
            _ax=axs[1, 2]
            divider = make_axes_locatable(_ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(_im, cax=cax, orientation='vertical')

            axs[1, 2].set_title('RGB Residual')
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
            plt.subplots_adjust(wspace=0, hspace=0)

            plt.savefig(
                f'{self.vis_dir}/{idx:05d}_{iter:04d}.png', bbox_inches='tight', dpi=300)

            plt.clf()

            if self.verbose:
                print(
                    f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
