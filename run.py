import argparse
import random

import numpy as np
import torch

from src import config
from src.O2V_SLAM import O2V_SLAM

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
    
def main():
    
    parser = argparse.ArgumentParser(
        description=' Arguments for O2V_SLAM'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()
    
    cfg = config.load_config(
        args.config,
        '/home/tiemuer/tiemuer/semantic-SLAM/configs/Repliac_ICCV.yaml'
    )
    
    
    slam = O2V_SLAM(cfg, args)
    slam.run()
    
    

if __name__ == '__main__':
    main()