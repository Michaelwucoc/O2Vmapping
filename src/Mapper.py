import os

os.environ['OMP_NUM_THREADS'] = '4'
import time

import cv2

cv2.setNumThreads(1)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples_tracking_semantic, get_tensor_from_camera, random_select,
                        get_samples, normalize_3d_coordinate, normalize_3d_coordinate_semantic)
from src.datasets import get_dataset

import src.datasets
import matplotlib.pyplot as plt


class Mapper(object):

    def __init__(self, cfg, args, slam):

        self.cfg = cfg
        self.args = args

        self.idx = slam.idx
        self.c = slam.shared_c
        self.bound = slam.bound

        self.logger = slam.logger
        self.output = slam.output
        self.renderer = slam.renderer
        self.decoders = slam.shared_decoders
        self.clipper = slam.clipper

        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.mapping_frist_frame = slam.mapping_first_frame

        self.estimate_c2w_list = slam.estimate_c2w_list

        self.scale = cfg['scale']
        self.device = cfg['mapping']['device']

        self.BA = False
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']

        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.pixels = cfg['mapping']['pixels']
        self.iters = cfg['mapping']['iters']
        self.every_frame = cfg['mapping']['every_frame']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.query_text = ["chair", "EXIT", "poster", "tap", "trash can", "white wire", "black shoes", "red book",
                           "window", "plotted plant"]
        self.query_text.insert(0, cfg['clip']['query_text'])
        self.sample_way = 'all'

        self.frustum_feature_selection = False

        self.keyframe_dict = []
        self.keyframe_list = []

        self.frame_reader = get_dataset(
            cfg,
            args,
            self.scale,
            devcie=self.device
        )
        self.n_img = len(self.frame_reader)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = \
            slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, num, N_samples=16, pixels=100):

        device = self.device
        H, W, fx, fy, cx, cy = \
            self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0,
            H,
            0,
            W,
            pixels,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            c2w,
            gt_depth,
            gt_color,
            device
        )

        gt_depth = gt_depth.reshape(-1, 1).repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)

        near = gt_depth * 0.8
        far = gt_depth + 0.5

        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (N_rays, N_samples, 3)
        vertices = pts.reshape(-1, 3).cpu().numpy()

        keyframe_list = []

        for id, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)

            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            home_vertices = np.concatenate([vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4, 1)
            cam_cord_homo = w2c @ home_vertices
            cam_cord = cam_cord_homo[:, :3]

            K = np.array(
                [[fx, .0, cx],
                 [.0, fy, cy],
                 [.0, .0, 1.0]]
            ).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K @ cam_cord
            z = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z
            uv = uv.astype(np.float32)

            edge = 20
            mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * (uv[:, 1] < H - edge) * (uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum() / uv.shape[0]
            keyframe_list.append(
                {
                    'id': id,
                    'percent_inside': percent_inside
                }
            )

        keyframe_list = sorted(
            keyframe_list,
            key=lambda i: i['percent_inside'],
            reverse=True
        )

        selected_keyframe_list = [
            dic['id'] for dic in keyframe_list if dic['percent_inside'] > 0.00
        ]
        selected_keyframe_list = list(
            np.random.permutation(
                np.array(selected_keyframe_list)
            )[:num]
        )
        return selected_keyframe_list

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):

        H, W, fx, fy, cx, cy, = \
            self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask

        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i + remap_chunk, 0],
                                 uv[i:i + remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * \
               (uv[:, 1] < H - edge) * (uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak - ray_o
        dist = torch.sum(dist * dist, axis=1)
        mask2 = dist < 0.5 * 0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def optimize_map(self, iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, keyframe_dict, keyframe_list, cur_c2w,
                     init):

        H, W, fx, fy, cx, cy = \
            self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c = self.c
        cfg = self.cfg
        device = self.device
        gt_depth_semantic = gt_depth.clone()

        bottom = torch.from_numpy(
            np.array([0, 0, 0, 1.]).reshape([1, 4])
        ).type(torch.float32).to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []

        else:
            num = self.mapping_window_size - 2
            optimize_frame = self.keyframe_selection_overlap(
                gt_color,
                gt_depth,
                cur_c2w,
                keyframe_dict[:-1],
                num
            )

        # add the last keyframe and the new one
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame += [len(keyframe_list) - 1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        # pixs_per_image = 1000 // len(optimize_frame)
        pixs_per_image = 1000

        color_decoders_para_list = []
        coarse_grid_para = []
        middle_grid_para = []
        fine_grid_para = []
        color_grid_para = []
        gt_depth_np = gt_depth.cpu().numpy()

        if self.frustum_feature_selection:
            masked_c_grad = {}
            mask_c2w = cur_c2w

        for key, val in c.items():

            if not self.frustum_feature_selection:
                val = Variable(val.to(device), requires_grad=True)
                c[key] = val
                if key == 'grid_coarse':
                    coarse_grid_para.append(val)
                elif key == 'grid_middle':
                    middle_grid_para.append(val)
                elif key == 'grid_fine':
                    fine_grid_para.append(val)
                elif key == 'grid_color':
                    color_grid_para.append(val)

            else:
                mask = self.get_mask_from_c2w(
                    mask_c2w,
                    key,
                    val.shape[2:],
                    gt_depth_np
                )
                mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).repeat(1, val.shape[1], 1, 1,
                                                                                                1)
                val = val.to(device)

                val_grad = val[mask].clone()
                val_grad = Variable(val_grad.to(device), requires_grad=True)

                masked_c_grad[key] = val_grad
                masked_c_grad[key + 'mask'] = mask

                if key == 'grid_coarse':
                    coarse_grid_para.append(val_grad)
                elif key == 'grid_middle':
                    middle_grid_para.append(val_grad)
                elif key == 'gird_fine':
                    fine_grid_para.append(val_grad)
                elif key == 'gird_color':
                    color_grid_para.append(val_grad)

        color_decoders_para_list += list(self.decoders.color_decoder.parameters())

        if self.BA:

            camera_tensor_list = []
            gt_camera_tensor_list = []

            for frame in optimize_frame:
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_c2w

                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)

                    gt_camera_tensor = get_tensor_from_camera(c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

            optimizer = torch.optim.Adam(
                [
                    {'params': color_decoders_para_list, 'lr': 0.0},
                    {'params': coarse_grid_para, 'lr': 0},
                    {'params': middle_grid_para, 'lr': 0},
                    {'params': fine_grid_para, 'lr': 0},
                    {'params': color_grid_para, 'lr': 0},
                    {'params': camera_tensor_list, 'lr': 0}
                ]
            )

        else:

            optimizer = torch.optim.Adam(
                [
                    {'params': color_decoders_para_list, 'lr': 0.0},
                    {'params': coarse_grid_para, 'lr': 0},
                    {'params': middle_grid_para, 'lr': 0},
                    {'params': fine_grid_para, 'lr': 0},
                    {'params': color_grid_para, 'lr': 0}
                ]
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        for iter in range(iters):

            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key + 'mask']
                    val = val.to(device)
                    val[mask] = val_grad
                    c[key] = val

            self.stage = 'coarse'

            if iter <= int(iters * 0.3):
                self.stage = 'middle'

            elif iter <= int(iters * 0.7):
                self.stage = 'fine'

            else:
                self.stage = 'color'

            optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['color_decoders_lr'] * lr_factor
            optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr'] * lr_factor
            optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr'] * lr_factor
            optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr'] * lr_factor
            optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr'] * lr_factor

            if self.BA:
                if self.stage == 'color':
                    optimizer.param_groups[5]['lr'] = self.BA_cam_lr

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            camera_tensor_id = 0

            optimize_frame = [-1]
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']


                else:
                    gt_depth = gt_depth.to(device)
                    gt_color = gt_color.to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                    0,
                    H,
                    0,
                    W,
                    pixs_per_image,
                    H,
                    W,
                    fx,
                    fy,
                    cx,
                    cy,
                    c2w,
                    gt_depth,
                    gt_color,
                    self.device,
                    return_ij=True
                )
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            with torch.no_grad():

                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth

            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            ret = self.renderer.render_batch_ray(
                c,
                self.decoders,
                batch_rays_d,
                batch_rays_o,
                device,
                self.stage,
                batch_gt_depth
            )

            depth, uncertainty, color, _ = ret

            depth_mask = (batch_gt_depth > 0)

            loss = torch.abs(
                batch_gt_depth[depth_mask] - depth[depth_mask]
            ).sum()

            if self.stage == 'color':
                color_loss = torch.abs(batch_gt_color - color).sum()
                color_loss *= 0.2
                loss += color_loss

            if (iter % 50 == 0):
                print('iter : ', iter, 'loss : ', loss.item(), 'stage : ', self.stage)

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            # renew feature grid
            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key + 'mask']
                    val = val.detach()
                    val[mask] = val_grad.clone().detach()
                    c[key] = val

        depth_re, uncertainty, color, semantic_re = self.renderer.render_img(
            c,
            self.decoders,
            cur_c2w,
            self.device,
            stage='semantic',
            gt_depth=gt_depth
        )

        y_coords, x_coords, pixels_num = self.clipper.choose_pixels(H, W, way='all')
        semantic_c2w = gt_c2w.clone().to(self.device)

        # depth_semantic = gt_depth_semantic
        depth_semantic = depth_re.to(gt_depth_semantic.dtype)
        depth_semantic = depth_semantic[y_coords, x_coords].reshape((-1))

        y_coords, x_coords = torch.from_numpy(y_coords).to(self.device), torch.from_numpy(x_coords).to(self.device)
        x_c = (x_coords - cx) * depth_semantic / fx
        y_c = (y_coords - cy) * depth_semantic / fy
        z_c = depth_semantic
        one_c = torch.ones(z_c.shape[0], device=self.device)
        points = torch.stack([x_c, y_c, z_c, one_c], dim=-1).to(self.device)

        semantic_c2w[:3, 1] *= -1
        semantic_c2w[:3, 2] *= -1

        points = points @ semantic_c2w.T

        print('=', points.shape)

        mask = (points[:, 0] < self.bound[0][1]) & (points[:, 0] > self.bound[0][0]) & \
               (points[:, 1] < self.bound[1][1]) & (points[:, 1] > self.bound[1][0]) & \
               (points[:, 2] < self.bound[2][1]) & (points[:, 2] > self.bound[2][0])

        print('==', points.shape)
        points = points[mask]
        features = self.c['grid_semantic']

        def trilinear_interpolation(tensor, point):
            m, n, q = tensor.shape[-3:]
            x, y, z = point[:, 0], point[:, 1], point[:, 2]

            x_floor, y_floor, z_floor = torch.floor(x).long(), torch.floor(y).long(), torch.floor(z).long()
            x_ceil, y_ceil, z_ceil = x_floor + 1, y_floor + 1, z_floor + 1

            x_floor = torch.clamp(x_floor, 0, m - 1)
            y_floor = torch.clamp(y_floor, 0, n - 1)
            z_floor = torch.clamp(z_floor, 0, q - 1)
            x_ceil = torch.clamp(x_ceil, 0, m - 1)
            y_ceil = torch.clamp(y_ceil, 0, n - 1)
            z_ceil = torch.clamp(z_ceil, 0, q - 1)

            xd = x - x_floor.float()
            yd = y - y_floor.float()
            zd = z - z_floor.float()

            wx = torch.stack([(1.0 - xd), xd], dim=1)
            wy = torch.stack([(1.0 - yd), yd], dim=1)
            wz = torch.stack([(1.0 - zd), zd], dim=1)

            batch_size = 100000
            values_list = []
            for i in range(0, point.size(0), batch_size):
                c000 = tensor[0, :, x_floor[i:i + batch_size], y_floor[i:i + batch_size], z_floor[i:i + batch_size]].T
                c001 = tensor[0, :, x_floor[i:i + batch_size], y_floor[i:i + batch_size], z_ceil[i:i + batch_size]].T
                c010 = tensor[0, :, x_floor[i:i + batch_size], y_ceil[i:i + batch_size], z_floor[i:i + batch_size]].T
                c011 = tensor[0, :, x_floor[i:i + batch_size], y_ceil[i:i + batch_size], z_ceil[i:i + batch_size]].T
                c100 = tensor[0, :, x_ceil[i:i + batch_size], y_floor[i:i + batch_size], z_floor[i:i + batch_size]].T
                c101 = tensor[0, :, x_ceil[i:i + batch_size], y_floor[i:i + batch_size], z_ceil[i:i + batch_size]].T
                c110 = tensor[0, :, x_ceil[i:i + batch_size], y_ceil[i:i + batch_size], z_floor[i:i + batch_size]].T
                c111 = tensor[0, :, x_ceil[i:i + batch_size], y_ceil[i:i + batch_size], z_ceil[i:i + batch_size]].T
                interpolated_values = (wx[i:i + batch_size, 0] * wy[i:i + batch_size, 0] * wz[i:i + batch_size,
                                                                                           0]).unsqueeze(-1) * c000
                interpolated_values += (wx[i:i + batch_size, 0] * wy[i:i + batch_size, 0] * wz[i:i + batch_size,
                                                                                            1]).unsqueeze(-1) * c001
                interpolated_values += (wx[i:i + batch_size, 0] * wy[i:i + batch_size, 1] * wz[i:i + batch_size,
                                                                                            0]).unsqueeze(-1) * c010
                interpolated_values += (wx[i:i + batch_size, 0] * wy[i:i + batch_size, 1] * wz[i:i + batch_size,
                                                                                            1]).unsqueeze(-1) * c011
                interpolated_values += (wx[i:i + batch_size, 1] * wy[i:i + batch_size, 0] * wz[i:i + batch_size,
                                                                                            0]).unsqueeze(-1) * c100
                interpolated_values += (wx[i:i + batch_size, 1] * wy[i:i + batch_size, 0] * wz[i:i + batch_size,
                                                                                            1]).unsqueeze(-1) * c101
                interpolated_values += (wx[i:i + batch_size, 1] * wy[i:i + batch_size, 1] * wz[i:i + batch_size,
                                                                                            0]).unsqueeze(-1) * c110
                interpolated_values += (wx[i:i + batch_size, 1] * wy[i:i + batch_size, 1] * wz[i:i + batch_size,
                                                                                            1]).unsqueeze(-1) * c111
                values_list.append(interpolated_values)

            values = torch.cat(values_list, dim=0)

            return values

        p_nor = normalize_3d_coordinate_semantic(points[:, :3].clone(), self.bound, features.shape[-3:])
        semantic = trilinear_interpolation(features, p_nor)

        text_probs = self.clipper.clip_feature_text(semantic, self.query_text)
        query_text_probs = text_probs[:, 0].reshape(H, W, -1)

        depth_np = depth_re.detach().cpu().numpy()
        semantic_np = query_text_probs.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()

        fig, axs = plt.subplots(3)
        fig.set_figwidth(5)
        fig.set_figheight(12)

        axs[0].imshow(depth_np)
        axs[0].set_title('depth')

        if not os.path.exists(f'{self.output}/{self.query_text[0]}/probsnp'):
            os.makedirs(f'{self.output}/{self.query_text[0]}/probsnp')
        np.save(f'{self.output}/{self.query_text[0]}/probsnp/probsnp_{idx:04d}.npy', semantic_np)

        if not os.path.exists(f'{self.output}/{self.query_text[0]}/semanticprobs'):
            os.makedirs(f'{self.output}/{self.query_text[0]}/semanticprobs')
        color_map = 'jet'
        semantics = (plt.cm.get_cmap(color_map)(semantic_np)[..., :3] * 255).reshape(-1, 3).reshape(H, W, 3).astype(
            np.uint8)
        cv2.imwrite(f'{self.output}/{self.query_text[0]}/semanticprobs/semanticprobs_{idx:04d}.png',
                    cv2.cvtColor(semantics, cv2.COLOR_BGR2RGB))

        axs[1].imshow(semantic_np, cmap='jet', vmin=0, vmax=1)
        axs[1].set_title('semantic_probs')

        # axs[1, 0].imshow(gt_depth_np)
        # axs[1, 0].set_title('gt_depth')

        # axs[1, 1].imshow(gt_semantic_np, cmap="tab20")
        # axs[1, 1].set_title('gt_semantic')

        axs[2].imshow(color_np)
        axs[2].set_title('color')

        # axs[1, 2].imshow(gt_color_np)
        # axs[1, 2].set_title('gt_color')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{self.output}/{self.query_text[0]}/ReplicaS_{idx:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
        plt.clf()

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None

    def build_semantic_map(self, idx, gt_color, gt_depth, gt_c2w):

        H, W, fx, fy, cx, cy = \
            self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c = self.c
        cfg = self.cfg
        device = self.device

        semantic_c2w = gt_c2w.clone()

        color = gt_color.clone()

        print(f'frame {idx} samming ...')
        masks = self.clipper.get_masks_from_tensor(color)
        masks_np = self.clipper.get_segment_masks(masks)

        print(f'frame {idx} clipping ...')
        clip_features = self.clipper.build_clip_index(masks_np, color)

        print(f'frame {idx} indexing ...')

        y_coords, x_coords, pixels_num = self.clipper.choose_pixels(H, W, way=self.sample_way)

        # 假设y_coords, x_coords 是 numpy
        y_coords = y_coords.astype(np.int64)
        x_coords = x_coords.astype(np.int64)

        # 转tensor并裁剪
        y_coords_t = torch.from_numpy(y_coords).to(gt_depth.device)
        x_coords_t = torch.from_numpy(x_coords).to(gt_depth.device)

        H, W = gt_depth.shape
        y_coords_t = torch.clamp(y_coords_t, 0, H - 1)
        x_coords_t = torch.clamp(x_coords_t, 0, W - 1)

        indexs = self.clipper.simple_feature_from_masks(masks_np, y_coords, x_coords)
        indexs = indexs.T

        depth = gt_depth.clone()
        depth_vals = depth[y_coords_t.long(), x_coords_t.long()]  # 用裁剪后的坐标索引

        # 1. 裁剪坐标，确保合法
        y_coords = np.clip(y_coords, 0, masks_np.shape[1] - 1).astype(np.int64)
        x_coords = np.clip(x_coords, 0, masks_np.shape[2] - 1).astype(np.int64)

        print(f"Clipped y_coords max: {y_coords.max()}, min: {y_coords.min()}")
        print(f"Clipped x_coords max: {x_coords.max()}, min: {x_coords.min()}")

        # 2. 转tensor，移动到设备
        y_coords_t = torch.from_numpy(y_coords).to(self.device)
        x_coords_t = torch.from_numpy(x_coords).to(self.device)

        # 3. 使用裁剪后的坐标索引depth
        depth_vals = gt_depth[y_coords_t, x_coords_t]  # shape: (num_points,)

        # 4. 断言长度是否一致：depth_vals长度 和 坐标长度一致
        assert depth_vals.shape[0] == y_coords_t.shape[0] == x_coords_t.shape[0], "depth和坐标长度不匹配"

        # 5. 转为float方便计算相机坐标
        depth_vals = depth_vals.float()

        # 6. 计算相机坐标
        x_coords_f = x_coords_t.float()
        y_coords_f = y_coords_t.float()

        x_c = (x_coords_f - cx) * depth_vals / fx
        y_c = (y_coords_f - cy) * depth_vals / fy
        z_c = depth_vals

        one_c = torch.ones_like(z_c, device=self.device)
        points = torch.stack([x_c, y_c, z_c, one_c], dim=-1)

        semantic_c2w[:3, 1] *= -1
        semantic_c2w[:3, 2] *= -1

        points = points @ semantic_c2w.T

        mask = (points[:, 0] < self.bound[0][1]) & (points[:, 0] > self.bound[0][0]) & \
               (points[:, 1] < self.bound[1][1]) & (points[:, 1] > self.bound[1][0]) & \
               (points[:, 2] < self.bound[2][1]) & (points[:, 2] > self.bound[2][0]) & \
               ((points[:, 0] > 0) | (points[:, 0] < 0) | (points[:, 1] > 0) | (points[:, 1] < 0) | (
                       points[:, 2] > 0) | (points[:, 2] < 0))

        points = points[mask]

        # ps, pnor (point_num , 3)
        features = self.c['grid_semantic']
        count = self.c['grid_count']
        p_nor = normalize_3d_coordinate_semantic(points[:, :3].clone(), self.bound, features.shape[-3:])
        p_nor = p_nor.round().long()

        print(f"p_nor shape: {p_nor.shape}")

        with torch.no_grad():
            feature = self.clipper.indexs_for_feature(indexs, clip_features).to(self.device)

            if p_nor.numel() > 0:
                count[p_nor[:, 0], p_nor[:, 1], p_nor[:, 2]] += 1
                new_w = 1. / (count[p_nor[:, 0], p_nor[:, 1], p_nor[:, 2]] + 1e-5)

                assert feature.shape[1] == p_nor.shape[0], "feature和p_nor长度不匹配"

                features[0, :, p_nor[:, 0], p_nor[:, 1], p_nor[:, 2]] = \
                    feature.T * new_w + features[0, :, p_nor[:, 0], p_nor[:, 1], p_nor[:, 2]] * (1 - new_w)
            else:
                print("Warning: p_nor为空，跳过更新features")

        return True

    def run(self):

        cfg = self.cfg
        is_semantic = cfg['semantic']['is_semantic']
        fre_semantic = cfg['clip']['fre_semantic']

        if is_semantic:
            idx, gt_color, gt_depth, gt_c2w, gt_semantic = \
                self.frame_reader[0]

        else:
            idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]

        self.estimate_c2w_list[0] = gt_c2w.cpu()

        init = True
        prev_idx = -1

        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img - 1:
                    break
                if idx % self.every_frame == 0 and idx != prev_idx:
                    break
                time.sleep(0.1)

            prev_idx = idx

            print(Fore.YELLOW)
            print('Mapping Frame :', idx.item())
            print(Style.RESET_ALL)

            idx, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            # build semantic-probs map
            if idx % fre_semantic == 0:
                self.build_semantic_map(idx, gt_color, gt_depth, gt_c2w)

            if not init:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
                outer_joint_iters = 1

            else:
                lr_factor = 5 * cfg['mapping']['lr_factor']
                iters = 500
                outer_joint_iters = 1

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            iters = iters // outer_joint_iters

            for outer_joint_iter in range(outer_joint_iters):
                # self.BA = len(self.keyframe_list) > 4
                _ = self.optimize_map(
                    iters,
                    lr_factor,
                    idx,
                    gt_color,
                    gt_depth,
                    gt_c2w,
                    self.keyframe_dict,
                    self.keyframe_list,
                    cur_c2w,
                    init=init
                )

                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                if outer_joint_iter == outer_joint_iters - 1:

                    if (idx % self.keyframe_every == 0 or (idx == self.n_img - 2)) and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append(
                            {
                                'gt_c2w': gt_c2w.cpu(),
                                'idx': idx,
                                'color': gt_color.cpu(),
                                'depth': gt_depth.cpu(),
                                'est_c2w': cur_c2w.clone()
                            }
                        )

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

            init = False
            self.mapping_frist_frame[0] = 1

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if idx == self.n_img - 1:
                break
