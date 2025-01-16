import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor


class Visualizer(object):
    
    def __init__(
        self, 
        freq,
        inside_freq,
        vis_dir,
        renderer,
        device = 'cuda:0'    
    ) -> None:
        
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.renderer = renderer
        self.inside_freq = inside_freq
        
        os.makedirs(f'{vis_dir}', exist_ok=True)
        
    def vis(
        self,
        idx, 
        iter,
        gt_depth,
        gt_color,
        c2w_or_camera_tensor,
        gt_semantic,
        c,
        decoders
    ):
        with torch.no_grad():
            
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                gt_semantic_np = gt_semantic.cpu().numpy()
                
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])
                    ).type(torch.float32).to(device)
                    c2w = get_camera_from_tensor(c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim = 0)
                    
                else :
                    c2w = c2w_or_camera_tensor
                    
                
                depth, uncertainty, color, semantic = self.renderer.render_img(
                    c, 
                    decoders,
                    c2w,
                    self.device,
                    stage = 'semantic',
                    gt_depth = gt_depth
                )                    