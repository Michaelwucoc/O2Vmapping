import os 

import torch

class Logger(object):
    
    def __init__(self, cfg, args, slam) -> None:
        
        self.ckptsdir = slam.ckptsdir
        
        self.gt_c2w_list = slam.gt_c2w_list
        self.estimate_c2w_list = slam.estimate_c2w_list
        
        self.shared_decoders = slam.shared_decoders
        self.shared_c = slam.shared_c
        
    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes = None):
        
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        print('Saving ckpt ...')
        
        torch.save(
            {
                'c': self.shared_c,
                'decoder_state_dict': self.shared_decoders.state_dict(),
                'gt_c2w_list': self.gt_c2w_list,
                'estimate_c2w_list': self.estimate_c2w_list,
                'keyframe_list': keyframe_list,
                'selected_keyframes': selected_keyframes,
                'idx' : idx,
                
            }, path, _use_new_zipfile_serialization=False
        )
        
        