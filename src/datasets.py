import os
import cv2
import glob

import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset


def get_dataset(cfg, args, scale, devcie='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=devcie)

def get_num_semantic_classes(path):
    
    semantic_list = sorted(glob.glob(path + '/semantic_class/semantic_class_*.png'), key = lambda file_name : int(file_name.split('_')[-1][:-4]))
    idxs = len(semantic_list)
    semantics = []
    
    for idx in range(idxs) : 
        semantic = cv2.imread(semantic_list[idx], cv2.IMREAD_UNCHANGED)
        semantics.append(semantic)
    
    semantic_classes = np.unique(
            np.unique(semantics)
    ).astype(np.uint8)        
    
    num_class = semantic_classes.shape[0]
    
    # need to remap to reduce vector lenth
    num = max(semantic_classes)
    return num

class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(BaseDataset, self).__init__()
        
        # 
        self.device = device
        self.scale = scale
        
        # ---cam---
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = \
            cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
        # ---semantic---    
        self.is_semantic = cfg['semantic']['is_semantic']
        
        # ---data---
        if args.input_folder is None : 
            self.input_folder = cfg['data']['input_folder']
        else : 
            self.input_folder = args.input_folder
            
    def __len__(self):
        return self.n_img
    
    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        if self.is_semantic : 
            semantic_path = self.semantic_paths[index]
            semantic_data = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            semantic_data = torch.from_numpy(semantic_data)
            
        color_data = cv2.imread(color_path)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
           
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        
        H, W = depth_data.shape
        
        
        color_data = cv2.resize(color_data, (W , H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale
        
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        
        
        if self.is_semantic : 
            return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device), semantic_data.to(self.device)
        else : 
            return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)

        
        
class ReplicaSemantic(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(ReplicaSemantic, self).__init__(cfg, args, scale, device)
        
        # self.color_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/rgb/rgb*.png'))
        # self.depth_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/depth/depth*.png'))
        # self.semantic_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/semantic_class/semantic_class_*.png'))
        
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/rgb/rgb*.png'), key = lambda x: int(x.split('/')[-1].strip('rgb_').strip('.png').strip('.jpg')) )
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/depth/depth*.png'), key = lambda x: int(x.split('/')[-1].strip('depth_').strip('.png').strip('.jpg')) )
        self.semantic_paths = sorted(glob.glob(f'{self.input_folder}/Sequence_2/semantic_class/semantic_class*.png'), key = lambda x: int(x.split('/')[-1].strip('semantic_class_').strip('.png').strip('.jpg')) )

        
        self.load_poses(f'{self.input_folder}/Sequence_2/traj_w_c.txt')
        self.n_img = len(self.color_paths)
        self.num_semantic_classes = get_num_semantic_classes(self.input_folder + '/Sequence_2/')
        
    def load_poses(self, path):
        self.poses = np.loadtxt(path, delimiter=" ").reshape(-1, 4, 4)
        self.poses = torch.from_numpy(self.poses)
        # y , z 
        self.poses[:, :3, 1] *= -1
        self.poses[:, :3, 2] *= -1
        
    def get_num_semantic_classes(path):
    
        semantic_list = sorted(glob.glob(path + '/Sequence_2/semantic_class/semantic_class_*.png'), key = lambda file_name : int(file_name.split('_')[-1][:-4]))
        idxs = len(semantic_list)
        semantics = []
        
        for idx in range(idxs) : 
            semantic = cv2.imread(semantic_list[idx], cv2.IMREAD_UNCHANGED)
            semantics.append(semantic)
        
        semantic_classes = np.unique(
                np.unique(semantic)
        ).astype(np.unit8)        
        
        
        return semantic_classes.shape[0]
    
    
class ReplicaClip(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(ReplicaClip, self).__init__(cfg, args, scale, device)
        
        self.color_paths =  sorted(glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/results/depth*.png'))

        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')
    
    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
        
    
class ScannetClip(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScannetClip, self).__init__(cfg, args, scale, device)
        # self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class OwndataClip(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(OwndataClip, self).__init__(cfg, args, scale, device)
        
        self.color_paths =  sorted(glob.glob(f'{self.input_folder}/rgbs/*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/depths/*.png'))

        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'poses'))
    
    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(os.listdir(path))
        for pose_path in pose_paths:
            pose_path = os.path.join(path, pose_path)
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)



dataset_dict = {
    "ReplicaSemantic" : ReplicaSemantic , 
    "ReplicaClip" : ReplicaClip,
    "scannetClip" : ScannetClip,
    "owndataClip" : OwndataClip
}
