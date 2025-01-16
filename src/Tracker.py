import copy 
import os
import time
from random import sample

import numpy as np
import torch
import cv2
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, get_samples_tracking_semantic,
                        get_sample_uv_semantic, get_rays_from_uv)

from src.datasets import get_dataset



class Tracker(object):
    def __init__(self, cfg, args, slam) -> None:
        
        self.cfg = cfg
        self.args = args
        
        
        self.scale = cfg['scale']
        
        self.idx = slam.idx
        self.bound = slam.bound
        self.output = slam.output
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.every_frame = slam.every_frame
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.results_dir = os.path.join(slam.output, 'tracking')
        
        self.device = cfg['mapping']['device']
        
        
        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg,
            args,
            self.scale,
            devcie = self.device
        )
        self.n_img = len(self.frame_reader)
        
        # windows : num_workers must be 0 ! 
        
        self.frame_loader = DataLoader(
            self.frame_reader,
            batch_size = 1,
            shuffle = False,
            num_workers = 0
        )
        
        
        
    
    
    def batchify(self, fn, netchunk):
        
        if netchunk == None:
            return fn
        
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + netchunk]) for i in range(0, inputs.shape[0], netchunk)], 0)
        
        return ret
    
    def updata_para_from_mapping(self):
        
        if self.mapping_idx[0] != self.prev_mapping_idx:
            print('updata the parameters from mapping...')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            
            self.prev_mapping_idx = self.mapping_idx[0].clone()
            
    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_gt_semantic = get_samples(
            0,
            H, 
            0,
            W,
            1000,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            c2w,
            gt_depth,
            gt_color,
            self.device
        )
        

        with torch.no_grad():

            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim = 1)
            inside_mask = t >= batch_gt_depth
                
        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]
        
        ret = self.renderer.render_batch_ray(
            self.c,
            self.decoders,
            batch_rays_d, 
            batch_rays_o,
            device = self.device,
            stage = 'color',
            gt_depth = batch_gt_depth
        )
        
        depth, uncertainty, color, semantic = ret
        
        mask = batch_gt_depth > 0
        
        loss = (torch.abs(batch_gt_depth - depth))[mask].sum()
        
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()      
    
    
    def select_ij(self, images_list, obi_list, obj_list, semantic_op = False):
        # (P)
        obi = torch.stack(obi_list, dim = 0).unsqueeze(-1)
        obj = torch.stack(obj_list, dim = 0).unsqueeze(-1)

        # repj_idx : (P, 1, 1000, 2)
        
        repj_idx = torch.cat([obi, obj], dim = -1).unsqueeze(1)
        repj_idx[..., 1] = repj_idx[..., 1] / self.H * 2.0 - 1.0
        repj_idx[..., 0] = repj_idx[..., 0] / self.W * 2.0 - 1.0
        
        # images : (P, H, W, 3) or (P, H, W, 1)
        if images_list[0].shape[-1] != 3:
            images = torch.stack(images_list, dim = 0).unsqueeze(-1)
        else :
            images = torch.stack(images_list, dim = 0)


        # repj_image : (5, 3, 1, 1000)
        repj_image = torch.nn.functional.grid_sample(
            images.permute(0, 3, 1, 2).float(),
            repj_idx,
            padding_mode='zeros'
        ).permute(0, 3, 1, 2).squeeze(-1)
        
        # return (5, 1000, 3)


        return repj_image


    def trans_coordinate(self, i, j, fx, fy, cx, cy, depth, c2w, w2c):
        # 1. 像素坐标转相机坐标
        cam_x = (i - cx) / fx * depth
        cam_y = (j - cy) / fy * depth
        cam_z = depth * torch.ones_like(cam_x)

        # 2. 相机坐标转世界坐标
        pc_cam = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_x)], dim=0)
        pc_world = torch.matmul(c2w.float(), pc_cam.view(4, -1).float())

        # 3. 世界坐标转相机坐标（假设 w2c 是 c2w 的逆矩阵）
        pc_cam_new = torch.matmul(w2c.float(), pc_world.float())
        
        # 4. 相机坐标转像素坐标
        new_x = (pc_cam_new[0] / pc_cam_new[2]) * fx + cx
        new_y = (pc_cam_new[1] / pc_cam_new[2]) * fy + cy

        new_x.requires_grad_()
        new_y.requires_grad_()

        new_i, new_j = torch.round(new_x), torch.round(new_y)

        return new_i, new_j

    def get_gird_feature(self, i, j, fx, fy, cx, cy, depth, c2w):

        # 1. 像素坐标转相机坐标
        cam_x = (i - cx) / fx * depth
        cam_y = (j - cy) / fy * depth
        cam_z = depth * torch.ones_like(cam_x)

        # 2. 相机坐标转世界坐标
        pc_cam = torch.stack([cam_x, cam_y, cam_z, torch.ones_like(cam_x)], dim=0)
        pc_world = torch.matmul(c2w.float(), pc_cam.view(4, -1).float())

        pc_world = pc_world[:3, :].transpose(1, 0)

        points = pc_world

        rets = self.renderer.eval_points(points, self.decoders, c = self.shared_c, stage = 'semantic', device = 'cuda:0')

        return rets

    def show_repj_pixels(image_list, i_list, j_list, radius = 5):


        fig, axs = plt.subplots(1, len(image_list), figsize=(20, 10))

        for i, pack in enumerate(zip(image_list, i_list, j_list)):

            image, opi, opj = pack
            image = image.detach().cpu().numpy()
            opi = opi.detach().cpu().numpy().astype(int)
            opj = opj.detach().cpu().numpy().astype(int)
            
            opi[opi >= 639] = 0
            opj[opj >= 479] = 0
            opi[opi <= 1] = 0
            opj[opj <= 1] = 0


            image[opj][opi] = (255, 255, 0)
            image[opj+1][opi] = (255, 255, 0)
            image[opj][opi+1] = (255, 255, 0)
            image[opj-1][opi] = (255, 255, 0)
            image[opj][opi-1] = (255, 255, 0)
            image[opj+1][opi+1] = (255, 255, 0)
            image[opj-1][opi-1] = (255, 255, 0)

            axs[i].imshow(image)
            axs[i].axis('off')

        plt.show()
        
        return True






    def semantic_BA(self, idxs, colors, depths, semantics, start_poses, gt_poses, iters = 150, init=False):
        '''
        
        '''
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device = self.device
        
        cam_last_row = torch.tensor([0, 0, 0, 1]).float().to(device)
        
        # 这里默认是升序的，没改poses和其他图像的顺序
        cam_idxs = sorted(idxs)
        # 一张图片就不用BA
        cam_conf = []
        if len(idxs) > 1 : 
            for i in range(0, len(cam_idxs)):
                # cam_conf.append(((len(cam_idxs) - 1 - i) / (len(cam_idxs) - 1)))
                cam_conf.append(1)
        else :
            return start_poses
        

        result_c2ws = []
        result_c2ws.append(start_poses[0])
        
        
            
        for ii, pkg in enumerate(zip(cam_idxs[:-1], start_poses[:-1])):
            
            idx, pose = pkg
            if pose.shape[1] != 4: 
                c2w = get_camera_from_tensor(pose)
            else :
                c2w = pose
            
            if c2w.shape[0] == 3 : 
                c2w = torch.cat([c2w, cam_last_row.view(1, 4)], dim = 0)

            
                

            # 当前关键帧sample出n条射线，返回其i, j， depth值
            rays_o, rays_d, sample_depth, sample_color, sample_semantic, sample_i, sample_j = get_samples_tracking_semantic(
                0, H, 0, W, 1000, H, W, fx, fy, cx, cy, c2w, depths[ii], colors[ii], semantics[ii], device, return_ij=True)
            
            with torch.no_grad():

                rays_o = rays_o.clone().detach().unsqueeze(-1)
                rays_d = rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(device) - rays_o) / rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim = 1)
                inside_mask = t >= sample_depth
                
            rays_d = rays_d[inside_mask]
            rays_o = rays_o[inside_mask]
            sample_depth = sample_depth[inside_mask]
            sample_color = sample_color[inside_mask]
            sample_semantic = sample_semantic[inside_mask]
            sample_i = sample_i[inside_mask]
            sample_j = sample_j[inside_mask]
            
            
            ret = self.renderer.render_batch_ray(
                self.c,
                self.decoders,
                rays_d, 
                rays_o,
                device = self.device,
                stage = 'color',
                gt_depth = sample_depth
            )
        
            depth_render, uncertainty, color_render, semantic_render = ret
            
            
            cam_var_list = []
            cam_var_list_gt = []
            
            # 相机位姿扔进去当前位姿以后的所有位姿
            for idx, pose, gt_pose in zip(idxs[ii+1:], start_poses[ii+1:], gt_poses[ii+1:]):
                cam_var = get_tensor_from_camera(pose.detach().cpu())
                cam_var = Variable(cam_var.to(device), requires_grad=True)
                cam_var_list.append(cam_var)

                cam_var_gt = get_tensor_from_camera(gt_pose.detach()).to(device)
                cam_var_list_gt.append(cam_var_gt)
                
                
            optim_list = []
            
            optim_list.append({"params": cam_var_list, "lr": 0.01})
            
            optimizer = torch.optim.AdamW(optim_list)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=False)



            for iter in range(iters):

                opi_list = []
                opj_list = []
                optim_pose = []
                depth_repj_list = []
                color_repj_list = []
                semantic_repj_list = []
                
                optimizer.zero_grad()


                for cam, cam_gt in zip(cam_var_list, cam_var_list_gt):
            
                    pose = get_camera_from_tensor(cam)
                    pose_row = torch.cat([pose, cam_last_row.view(1, 4)], dim = 0)
                    w2c = torch.inverse(pose_row)
                    new_i, new_j = self.trans_coordinate(sample_i, sample_j, fx, fy, cx, cy, sample_depth, c2w, w2c)
                    # print('old : ', sample_i[0].item(), sample_j[0].item(),'new :', new_i[0].item(), new_j[0].item())
                    rays_o_repj, rays_d_repj = get_rays_from_uv(new_i, new_j, pose_row, H, W, fx, fy, cx, cy, device)
                    
                    with torch.no_grad():

                        rays_o = rays_o.clone().detach().unsqueeze(-1)
                        rays_d = rays_d.clone().detach().unsqueeze(-1)
                        t = (self.bound.unsqueeze(0).to(device) - rays_o) / rays_d
                        t, _ = torch.min(torch.max(t, dim=2)[0], dim = 1)
                        inside_mask = t >= sample_depth
                        
                    rays_d = rays_d[inside_mask]
                    rays_o = rays_o[inside_mask]
                    sample_depth = sample_depth[inside_mask]
                    sample_color = sample_color[inside_mask]
                    sample_semantic = sample_semantic[inside_mask]
                    sample_i = sample_i[inside_mask]
                    sample_j = sample_j[inside_mask]
                    
                    
                    ret_repj = self.renderer.render_batch_ray(
                        self.c,
                        self.decoders,
                        rays_d_repj, 
                        rays_o_repj,
                        device = self.device,
                        stage = 'color',
                        gt_depth = sample_depth
                    )
                    
                    depth_render, uncertainty, color_render, semantic_render = ret_repj
                    
                    opi_list.append(new_i)
                    opj_list.append(new_j)
                    optim_pose.append(pose)

                
                
                depths_list = []
                colors_list = []
                semantics_list = []
                

                for idx, optim_pose_, opi, opj in zip(idxs[ii+1:], optim_pose, opi_list, opj_list) :
                    
                    # batch_rays_o, batch_rays_d = get_rays_from_uv(opi, opj, c2w, H, W, fx, fy, cx, cy, device)
                    # 现在是读render好的图像结果
                    ret = colors_list[idx], depths_list[idx], semantics_list[idx]
            
                    color, depth, semantic = ret
                    depths_list.append(torch.from_numpy(depth).to(device))
                    colors_list.append(torch.from_numpy(color).to(device))
                    semantics_list.append(torch.from_numpy(semantic).to(device))

                                
                loss = 0


                # ---------------color ---------------
                # color_repj = select_ij(colors_list, opi_list, opj_list)
                # color_gt = sample_color.unsqueeze(0).repeat(len(color_repj), 1, 1)

                # mask = torch.sum(color_repj, dim = -1) > 0
                # color_repj = color_repj[mask]
                # color_gt = color_gt[mask]
                # color_loss = abs(color_gt - color_repj).sum()
                
                # ---------------depth ---------------            
                # depth_gt = select_ij(depths_list, opi_list_gt, opj_list_gt)
                # depth_render = select_ij(depths_list, opi_list, opj_list)
                # mask = (depth_render > 0) & (depth_gt > 0)
                # depth_loss = abs(depth_gt[mask] - depth_render[mask]).sum()

                # ---------------semantic---------------
                # semantic_repj [p, 1000] ,semantic_gt [p, 1000]

                semantic_repj = self.select_ij(semantics_list, opi_list, opj_list).squeeze(-1)
                semantic_gt = logits.unsqueeze(0).repeat(len(semantics_list), 1, 1)


                semantic_repj = semantic_repj.view(-1)
                semantic_gt = semantic_gt.view(len(semantic_repj), -1)

                
                
                loss += semantic_loss
                
                
                if iter % 10 == 0 :
                    s_poses = torch.stack(cam_var_list, dim = 0).detach().clone()
                    g_poses = []
                    gg_poses = gt_poses[ii+1:]
                    for dd in range(len(gg_poses)):
                        g_poses.append(get_tensor_from_camera(gg_poses[dd]))
                        
                    g_poses = torch.stack(g_poses, dim = 0).to(device)

                    print('iter : ',iter, 'BA pose loss : ', abs(s_poses - g_poses).sum().item(), 'loss : ', loss.item())
                
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                
                    
            pose = cam_var_list[0]
            c2w = get_camera_from_tensor(pose).detach().clone()
            if c2w.shape[0] == 3:
                c2w = torch.cat([c2w, cam_last_row.view(1, 4)], dim = 0)

            result_c2ws.append(c2w)
            
            # with open(render_path + '/poses_op.txt', 'w') as file :
            #     c2w_list = result_c2ws
            #     for matrix in c2w_list :
            #         # kitti数据集格式,仅用转移矩阵前三行
            #         matrix = matrix[:3].detach()
            #         matrix_str = ' '.join(str(x.item()) for x in matrix.flatten())
            #         file.write(matrix_str + '\n')

            # with open(render_path + '/poses_gt.txt', 'w') as file :
            #     c2w_list = gt_poses
            #     for matrix in c2w_list :
            #         # kitti数据集格式,仅用转移矩阵前三行
            #         matrix = matrix[:3].detach()
            #         matrix_str = ' '.join(str(x.item()) for x in matrix.flatten())
            #         file.write(matrix_str + '\n')
            

                
            
        return result_c2ws
            
            
                
                

        

    def run(self):
        
        
        
        self.keyframe_color = []
        self.keyframe_depth = []
        self.keyframe_semantic = []
        self.keyframe_idx = []
        self.keyframe_gt_pose = []
        self.keyframe_estimate_pose = []
        
        
        device = self.device
        self.c = {}
        
        
        pbar = self.frame_loader
        
        
        for data in pbar:
            
            idx, gt_color, gt_depth, gt_c2w = data
                        
            
            print(Fore.CYAN)
            print("Tracking Frame : ", idx[0].item())
            print(Style.RESET_ALL)
            
            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]            
            
            
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                
                while (self.mapping_idx[0] != idx - 1):
                    time.sleep(0.1)
                pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            
            
            # self.updata_para_from_mapping()
            
            if idx <= 1 : 
                c2w = gt_c2w
                
            # else :
            #     # c2w : 4 x 4 matrx
            #     # tensor : [a, b, c, w, x, ,y, z]
            #     gt_camera_tensor = get_tensor_from_camera(gt_c2w)
            #     pre_c2w = pre_c2w.float()
            #     delta = pre_c2w @ self.estimate_c2w_list[idx - 2].to(device).float().inverse()
            #     estimated_new_cam_c2w = delta  @  pre_c2w
                
            #     camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())
                
            #     camera_tensor = Variable(camera_tensor.to(device), requires_grad = True)
            #     cam_para_list = [camera_tensor]
                
            #     optimizer_camera = torch.optim.Adam(cam_para_list, lr = 0.001)
                
                
            #     initial_loss_camera_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()
                
            #     candidate_cam_tensor = None
            #     current_min_loss = 100000000.
                
            #     num_cam_iters = 10
            #     for cam_iter in range(num_cam_iters):
                    
            #         camera_tensor.requires_grad = True
                    
            #         loss = self.optimize_cam_in_batch(
            #             camera_tensor,
            #             gt_color,
            #             gt_depth,
            #             gt_semantic,
            #             1000,
            #             optimizer_camera
            #         )
                    
            #         if cam_iter == 0:
            #             initial_loss = loss
                        
            #         loss_camera_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()
                    
            #         print(
            #             f'Rerendering loss:{initial_loss:.2f}->{loss:.2f}' + 
            #             f'camera tensor error:{initial_loss_camera_tensor:.4f}->{loss_camera_tensor}'
            #         )
                    
            #         if loss < current_min_loss:
            #             current_min_loss = loss
            #             candidate_cam_tensor = camera_tensor.clone().detach()
                        
                
            #     bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(self.device)
                
            #     c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
            #     c2w = torch.cat([c2w, bottom], dim = 0)
                
            
            self.estimate_c2w_list[idx] = gt_c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
                      
            
            self.idx[0] = idx
            
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                
                
            # with open(self.results_dir + '_gt.txt', 'w') as file :
            #     c2w_list = self.gt_c2w_list
            #     for matrix in c2w_list :
            #         # kitti数据集格式,仅用转移矩阵前三行
            #         matrix = matrix[:3].detach()
            #         matrix_str = ' '.join(str(x.item()) for x in matrix.flatten())
            #         file.write(matrix_str + '\n')
                
            # with open(self.results_dir + '_semantic_tracking.txt', 'w') as file : 
            #     c2w_list = self.estimate_c2w_list
            #     for matrix in c2w_list :
            #         # kitti数据集格式，仅用转移矩阵前三行 
            #         matrix = matrix[:3].detach()
            #         matrix_str = ' '.join(str(x.item()) for x in matrix.flatten())
            #         file.write(matrix_str + '\n')  
            
            
        
        print('转移矩阵已经写入文件:', self.results_dir)
        
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            
        
        