import os
import time 

import numpy as np

import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.datasets import get_dataset
from src.Logger import Logger
from src.Renderer import Renderer
from src.Clipper import Clipper


torch.multiprocessing.set_sharing_strategy('file_system')

class O2V_SLAM():
    
    def __init__(self, cfg, args):
        
        self.cfg = cfg
        self.args = args
              
        if args.output is None :
            self.output = cfg['data']['output_folder']
        else : 
            self.output = args.output
            
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = \
            cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
            

        model = config.get_model(cfg)
        self.shared_decoders = model
        
        self.scale = cfg['scale']
        
        self.load_bound(cfg)
        self.load_pretrain(cfg)
        self.grid_init(cfg)
        
        # 
        try:
            mp.set_start_method('spawn', force = True)
        except RuntimeError:
            pass
        
        self.frame_reader = get_dataset(cfg, args, self.scale)
        
        self.n_img = len(self.frame_reader)
        
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        
        self.mapping_cnt = torch.zeros((1)).int()
        self.mapping_cnt.share_memory_()
        
        self.every_frame = self.cfg['mapping']['every_frame']
        
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val
            
        self.shared_decoders = self.shared_decoders.to(self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        
        self.clipper = Clipper(cfg, args, self)
        self.renderer = Renderer(cfg, args, self)
        self.logger = Logger(cfg, args, self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        
        print('O2V-mapping init ...')
        
        
        
        
        
        
    def load_bound(self, cfg):
        
        self.bound = torch.from_numpy(
            np.array(cfg['mapping']['bound']) * self.scale
        )

        
        bound_divisible = cfg['grid_len']['bound_divisible']
        
        self.bound[:, 1] = (((self.bound[:, 1] - self.bound[:, 0]) / bound_divisible).int() + 1 ) * bound_divisible + self.bound[:, 0]
        
        self.shared_decoders.bound = self.bound
        self.shared_decoders.middle_decoder.bound = self.bound
        self.shared_decoders.fine_decoder.bound = self.bound
        self.shared_decoders.color_decoder.bound = self.bound
        self.shared_decoders.coarse_decoder.bound = self.bound
        # self.shared_decoders.semantic_decoder.bound = self.bound
        
        
    def load_pretrain(self, cfg):
        
        
        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'], map_location = cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        
        for key, val, in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
                    
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)
        
        
    def grid_init(self, cfg):
        
        coarse_grid_len = cfg['grid_len']['coarse']
        self.coarse_grid_len = coarse_grid_len
        
        middle_grid_len = cfg['grid_len']['middle']
        fine_grid_len = cfg['grid_len']['fine']
        color_grid_len = cfg['grid_len']['color']
        semantic_grid_len = cfg['grid_len']['semantic']
        
        self.middle_grid_len = middle_grid_len
        self.fine_grid_len = fine_grid_len
        self.color_grid_len = color_grid_len
        self.semantic_grid_len = semantic_grid_len
        
        
        c = {}
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1] - self.bound[:, 0]
        
        
        coarse_val_shape = list(
            map(int, (xyz_len /coarse_grid_len).tolist())
        )
        coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]
        self.coarse_val_shape = coarse_val_shape
        val_shape = [1, c_dim, *coarse_val_shape]
        coarse_val = torch.zeros(val_shape).normal_(mean=0, std = 0.01)
        c['grid_coarse'] = coarse_val
        
        
        middle_val_shape = list(
            map(int, (xyz_len /middle_grid_len).tolist())
        )
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std = 0.01)
        c['grid_middle'] = middle_val
        
        fine_val_shape = list(
            map(int, (xyz_len /fine_grid_len).tolist())
        )
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std = 0.01)
        c['grid_fine'] = fine_val
        
        
        color_val_shape = list(
            map(int, (xyz_len /color_grid_len).tolist())
        )
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std = 0.01)
        c['grid_color'] = color_val
        
        semantic_val_shape = list(
            map(int, (xyz_len /semantic_grid_len).tolist())
        )
        semantic_val_shape[0], semantic_val_shape[2] = semantic_val_shape[2], semantic_val_shape[0]
        self.semantic_val_shape = semantic_val_shape
        val_shape = [1, 512, *semantic_val_shape]
        semantic_val = torch.zeros(val_shape)
        c['grid_semantic'] = semantic_val
        
        
        semantic_count = list(
            map(int, (xyz_len /semantic_grid_len).tolist())
        )
        semantic_count[0], semantic_count[2] = semantic_count[2], semantic_count[0]
        self.semantic_count = semantic_count
        val_shape = [*semantic_val_shape]
        semantic_val = torch.zeros(val_shape)
        c['grid_count'] = semantic_val
        
        self.shared_c = c
        
    def tracking(self):
        while True : 
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)
        self.tracker.run()
        
    def mapping(self):
        self.mapper.run()
        
        
    def run(self):
        
        processes = []
        
        for i in range(2):
            if i == 0 :
                p = mp.Process(
                    target = self.tracking,
                    args = ()
                )
            elif i == 1 : 
                p = mp.Process(
                    target = self.mapping,
                    args = ()
                )
            p.start()
            processes.append(p)
            
        for p in processes :
            p.join()


if __name__ == '__main__':
    pass