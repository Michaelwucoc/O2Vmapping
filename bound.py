import torch
import numpy as np
import argparse

from src.config import load_config
from src.datasets import get_dataset


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device) # x_c/z_c, y_c/z_c
    dirs = dirs.reshape(-1, 1, 3)

    
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape) #世界坐标系
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1) # i.shape=600 * 440 = 264000
    j = j.reshape(-1) 
    indices = torch.randint(i.shape[0], (n,), device=device) # (1, 5,..24299)共1000个数
    indices = indices.clamp(0, i.shape[0])
    i = i
    j = j
    # (i,j)都是<(h,w)的格点坐标
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth
    color = color
    return i, j, depth, color


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    # i, j = torch.meshgrid(torch.linspace(
    #     W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    # i = i.t()  # transpose
    # j = j.t()

    i=torch.tensor([[0,H1-15],[0,W1-15]])
    j=torch.tensor([[0,H1-15],[0,W1-15]])

    i, j, depth, color = select_uv(i, j, 4, depth, color, device=device)
    return i, j, depth, color


def get_samples(H0, H1, W0, W1, H, W, fx, fy, cx, cy, c2w, depth, color, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    i, j, sample_depth, sample_color = get_sample_uv(H0, H1, W0, W1,4, depth, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color


def main(start_num, end_num):
    
    
    parser = argparse.ArgumentParser(
        description=' Arguments for Semantic_slam'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()
    
    cfg = load_config(
        args.config,
        '/home/tiemuer/semantic-SLAM/configs/owndata.yaml'
    )
    
    SCALE = 1
    frame_reader = get_dataset(cfg, args, scale=SCALE, devcie= 'cpu')

    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']


    pts_min_list, pts_max_list = [], []
    for idx in range(start_num,end_num):
        _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
        # t_min = gt_depth.min()
        t_max =  gt_depth.max()
        t_min = 0.01 * gt_depth.min()
        c2w = gt_c2w

        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(0, H, 0, W, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, 'cpu')
        # batch_rays_o.shape = (1000,3), batch_rays_d.shape = (1000,3)
        # batch_rays_o就是原始的读出来的东西

        all_pts_max = t_max * batch_rays_d +  batch_rays_o
        pts_min = torch.min(all_pts_max, dim=0)[0]
        pts_max = torch.max(all_pts_max, dim=0)[0]
        pts_min_list.append(pts_min.unsqueeze(0))
        pts_max_list.append(pts_max.unsqueeze(0))

        all_pts_min = t_min * batch_rays_d +  batch_rays_o
        pts_min = torch.min(all_pts_min, dim=0)[0]
        pts_max = torch.max(all_pts_min, dim=0)[0]
        pts_min_list.append(pts_min.unsqueeze(0))
        pts_max_list.append(pts_max.unsqueeze(0))


        # pts_min_list.append(batch_rays_o[0,:].unsqueeze(0))
        # pts_max_list.append(batch_rays_o[0,:].unsqueeze(0))

    pts_min_tensor = torch.cat(pts_min_list)
    pts_max_tensor = torch.cat(pts_max_list)

    all_pts_min = torch.min(pts_min_tensor, dim=0)[0]
    all_pts_max = torch.max(pts_max_tensor, dim=0)[0]

    return all_pts_min, all_pts_max


all_pts_min, all_pts_max = main(0,102)
print('min: ', all_pts_min)
print('max: ', all_pts_max)