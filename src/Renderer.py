import torch
import numpy as np
from src.common import get_rays, raw2outputs_nerf_color, sample_pdf, normalize_3d_coordinate, normalize_3d_coordinate_semantic
from collections import defaultdict
from torch.distributions.normal import Normal

import torch.nn.functional as F


class Renderer(object):
    
    def __init__(
        self, 
        cfg, 
        args, 
        slam, 
        points_batch_size = 100000,
        ray_batch_size = 100000
    ) -> None:
        
        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size
        
        self.lindisp = cfg['rendering']['lindisp']
        self.perturb = cfg['rendering']['perturb']
        self.N_samples = cfg['rendering']['N_samples']
        self.N_surface = cfg['rendering']['N_surface']
        self.N_importance = cfg['rendering']['N_importance']
        self.device = cfg['rendering']['device']
        
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        
        self.bound = slam.bound
        self.shared_decoders = slam.shared_decoders
        self.shared_c = slam.shared_c
        
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = \
            slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
            
        
        self.clipper = slam.clipper    
        

    def eval_points(self, points, decoders, c = None, stage = 'color', device = 'cuda:0'):
        
        '''
            input : points
            output : decoder_description(like density / color / semantic)
        '''
        
        points_split = torch.split(points, self.points_batch_size)
        bound = self.bound
        
        decoder_rets = []
        
        for points in points_split:
            
            mask = (points[:, 0] < bound[0][1]) & (points[:, 0] > bound[0][0]) & \
                   (points[:, 1] < bound[1][1]) & (points[:, 1] > bound[1][0]) & \
                   (points[:, 2] < bound[2][1]) & (points[:, 2] > bound[2][0])
                   
            points = points.unsqueeze(0)
            
            if stage == 'semantic':
                
                features = c['grid_semantic']
                
                def trilinear_interpolation(tensor, point):
                            m, n, q = tensor.shape[-3:]
                            x, y, z = point[:, 0], point[:, 1], point[:, 2]

                            # 取整数部分
                            x_floor, y_floor, z_floor = torch.floor(x).long(), torch.floor(y).long(), torch.floor(z).long()
                            x_ceil, y_ceil, z_ceil = x_floor + 1, y_floor + 1, z_floor + 1

                            # 边界处理
                            x_floor = torch.clamp(x_floor, 0, m - 1)
                            y_floor = torch.clamp(y_floor, 0, n - 1)
                            z_floor = torch.clamp(z_floor, 0, q - 1)
                            x_ceil = torch.clamp(x_ceil, 0, m - 1)
                            y_ceil = torch.clamp(y_ceil, 0, n - 1)
                            z_ceil = torch.clamp(z_ceil, 0, q - 1)

                            # 计算权重
                            xd = x - x_floor.float()
                            yd = y - y_floor.float()
                            zd = z - z_floor.float()
                            
                            wx = torch.stack([(1.0 - xd), xd], dim=1)
                            wy = torch.stack([(1.0 - yd), yd], dim=1)
                            wz = torch.stack([(1.0 - zd), zd], dim=1)

                            # 获取八个相邻点的值
                            # c000 = tensor[0, :, x_floor, y_floor, z_floor].T
                            # c001 = tensor[0, :, x_floor, y_floor, z_ceil].T
                            # c010 = tensor[0, :, x_floor, y_ceil, z_floor].T
                            # c011 = tensor[0, :, x_floor, y_ceil, z_ceil].T
                            # c100 = tensor[0, :, x_ceil, y_floor, z_floor].T
                            # c101 = tensor[0, :, x_ceil, y_floor, z_ceil].T
                            # c110 = tensor[0, :, x_ceil, y_ceil, z_floor].T
                            # c111 = tensor[0, :, x_ceil, y_ceil, z_ceil].T

                            # 进行插值
                            batch_size = 100000
                            values_list = []
                            for i in range(0, point.size(0), batch_size):
                                c000 = tensor[0, :, x_floor[i:i+batch_size], y_floor[i:i+batch_size], z_floor[i:i+batch_size]].T
                                c001 = tensor[0, :, x_floor[i:i+batch_size], y_floor[i:i+batch_size], z_ceil[i:i+batch_size]].T
                                c010 = tensor[0, :, x_floor[i:i+batch_size], y_ceil[i:i+batch_size], z_floor[i:i+batch_size]].T
                                c011 = tensor[0, :, x_floor[i:i+batch_size], y_ceil[i:i+batch_size], z_ceil[i:i+batch_size]].T
                                c100 = tensor[0, :, x_ceil[i:i+batch_size], y_floor[i:i+batch_size], z_floor[i:i+batch_size]].T
                                c101 = tensor[0, :, x_ceil[i:i+batch_size], y_floor[i:i+batch_size], z_ceil[i:i+batch_size]].T
                                c110 = tensor[0, :, x_ceil[i:i+batch_size], y_ceil[i:i+batch_size], z_floor[i:i+batch_size]].T
                                c111 = tensor[0, :, x_ceil[i:i+batch_size], y_ceil[i:i+batch_size], z_ceil[i:i+batch_size]].T
                                interpolated_values = (wx[i:i+batch_size, 0] * wy[i:i+batch_size, 0] * wz[i:i+batch_size, 0]).unsqueeze(-1) * c000
                                interpolated_values += (wx[i:i+batch_size, 0] * wy[i:i+batch_size, 0] * wz[i:i+batch_size, 1]).unsqueeze(-1) * c001
                                interpolated_values += (wx[i:i+batch_size, 0] * wy[i:i+batch_size, 1] * wz[i:i+batch_size, 0]).unsqueeze(-1) * c010
                                interpolated_values += (wx[i:i+batch_size, 0] * wy[i:i+batch_size, 1] * wz[i:i+batch_size, 1]).unsqueeze(-1) * c011
                                interpolated_values += (wx[i:i+batch_size, 1] * wy[i:i+batch_size, 0] * wz[i:i+batch_size, 0]).unsqueeze(-1) * c100
                                interpolated_values += (wx[i:i+batch_size, 1] * wy[i:i+batch_size, 0] * wz[i:i+batch_size, 1]).unsqueeze(-1) * c101
                                interpolated_values += (wx[i:i+batch_size, 1] * wy[i:i+batch_size, 1] * wz[i:i+batch_size, 0]).unsqueeze(-1) * c110
                                interpolated_values += (wx[i:i+batch_size, 1] * wy[i:i+batch_size, 1] * wz[i:i+batch_size, 1]).unsqueeze(-1) * c111
                                values_list.append(interpolated_values)

                            values = torch.cat(values_list, dim = 0)
                            
                            return values
                        
            
                p_nor = normalize_3d_coordinate_semantic(points.clone(), self.bound, features.shape[-3:])
                decoder_ret = trilinear_interpolation(features, p_nor)
                
                # features = c['grid_semantic']
                # p_nor = normalize_3d_coordinate_semantic(points.clone(), self.bound, features.shape[-3:])
                # p_nor = p_nor.round().long()
                # decoder_ret = features[0, :, p_nor[:, 0], p_nor[:, 1], p_nor[:, 2]].T
                
                
                
            else :
                
                decoder_ret = decoders(points, c_grid = c, stage = stage)
                
            decoder_ret.squeeze(0)

            
            if stage == 'color':
                decoder_ret[~mask, 3] = 100.0
                
            decoder_rets.append(decoder_ret)
            
        decoder_rets = torch.cat(decoder_rets, dim=0)
        
        return decoder_rets
    
    
    
    
    def render_batch_ray(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):
        
        
            '''
                return (N_rays, 1/3/nums_classes) description
            '''
        
            N_samples = self.N_samples
            N_surface = self.N_surface
            N_importance = self.N_importance
            
            N_rays = rays_o.shape[0]
            
            if gt_depth is None : 
                N_surface = 0
                near = 0.01
                
            else : 
                gt_depth = gt_depth.reshape(-1, 1)
                gt_depth_samples = gt_depth.repeat(1, N_samples)
                near = gt_depth_samples * 0.01
                
                
            with torch.no_grad():
                det_rays_o = rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                far_val, far_index = torch.min(torch.max(t, dim = 2)[0], dim = 1)
                far_val = far_val.unsqueeze(-1)
                far_val += 0.01
                
            if gt_depth is not None : 
                far = torch.clamp(far_val, 0, torch.max(gt_depth * 1.2))
                
            else :
                far = far_val
                
            
            if N_surface > 0 : 
                gt_none_zero_mask = gt_depth > 0    # (N)
                gt_none_zero = gt_depth[gt_none_zero_mask]  # (N)
                gt_none_zero = gt_none_zero.unsqueeze(-1)   # (N, 1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)    # (N, N_surface) 
                
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                
                # # use Guss
                
                # stddev = 0.02 * torch.ones(N_surface).to(device)
                # z_vals_surface_depth_none_zero = Normal(gt_none_zero.repeat(1, N_surface), stddev).sample().double()
                
                
                # use even
                
                t_vals_surface = torch.linspace(0., 1., steps = N_surface).double().to(device)
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * (1.-t_vals_surface) + 1.05*gt_depth_surface * (t_vals_surface)
                
                # depth points
                
                z_vals_surface = torch.zeros(gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero
                
                # no depth points
                
                z_vals_surface_depth_zero = near_surface * (1. - t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)    # no depth points
                
                z_vals_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero
                
        
            t_vals = torch.linspace(0., 1., steps = N_samples, device = device)
            z_vals = near * (1. - t_vals) + far * (t_vals)
            
            if self.perturb > 0. : 
                # random sample
                mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                
                t_rand = torch.rand(z_vals.shape).to(device)
                z_vals = lower + (upper - lower) * t_rand
                
            if N_surface > 0 : 
                z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_surface.double()], -1), -1)
                
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pointsf = pts.reshape(-1, 3)
                            
                
            
            descp = self.eval_points(pointsf, decoders, c, stage = stage if stage != 'semantic' else 'color', device = device)
            descp = descp.reshape(N_rays, N_samples + N_surface, -1)
            
            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                descp,
                z_vals,
                rays_d,
                occupancy=self.occupancy,
                device=device
            )
            
            if stage == 'semantic':
                
                pts = rays_o[..., None, :] + rays_d[..., None, :] * gt_depth[..., :, None]
                pointsf = pts.reshape(-1, 3)
                descp_semantic = self.eval_points(pointsf, decoders, c, stage = 'semantic', device = device)
                
                return depth, uncertainty, color, descp_semantic
            
            
            # 让他返回每个像素射线的clip feature
            # if stage == 'semantic':
            #     del descp
            #     descp_semantic = self.eval_points(pointsf, decoders, c, stage = 'semantic', device = device)
            #     descp_semantic = descp_semantic.reshape(N_rays, N_samples + N_surface, -1)
            #     semantic = torch.sum(weights[..., None] * descp_semantic, -2) # (N_rays, nums_classes)
                
            #     return depth, uncertainty, color, semantic
            
            if N_importance > 0:
                # importance sample
                z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det = (self.perturb == 0), device=device).detach()
                z_vals = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                
                pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
                pts = pts.reshape(-1, 3)
                
                descp = self.eval_points(pts, decoders, c, stage, device = device).reshape(N_rays, N_samples + N_importance + N_surface, -1)
                
                # importance sampling also used same strategy
    
                depth, uncertainty, color, weights = raw2outputs_nerf_color(
                    descp,
                    z_vals,
                    rays_d,
                    occupancy=self.occupancy,
                    device=device
                )
                
                if stage == 'semantic':
                    del descp
                    descp_semantic = self.eval_points(pointsf, decoders, c, stage = 'semantic', device = device)
                    descp_semantic = descp_semantic.reshape(N_rays, N_samples + N_surface, -1)
                    semantic = torch.sum(weights[..., None] * descp_semantic, -2) # (N_rays, nums_classes)
                    
                    return depth, uncertainty, color, semantic
                
            return depth, uncertainty, color, weights
        



    def render_batch_ray_nice(self, c, decoders, rays_d, rays_o, device, stage, gt_depth=None):

        """
        Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor): rendered depth.
            uncertainty (tensor): rendered uncertainty.
            color (tensor): rendered color.
        """

        
        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance
        

        N_rays = rays_o.shape[0]

        if stage == 'coarse':
            gt_depth = None
        if gt_depth is None:
            N_surface = 0
            near = 0.01
        else:
            gt_depth = gt_depth.reshape(-1, 1)
            gt_depth_samples = gt_depth.repeat(1, N_samples)
            near = gt_depth_samples*0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) -
                 det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        if gt_depth is not None:
            # in case the bound is too large
            far = torch.clamp(far_bb, 0,  torch.max(gt_depth*1.2))
        else:
            far = far_bb
        if N_surface > 0:
            if False:
                # this naive implementation downgrades performance
                gt_depth_surface = gt_depth.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).to(device)
                z_vals_surface = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
            else:
                # since we want to colorize even on regions with no depth sensor readings,
                # meaning colorize on interpolated geometry region,
                # we sample all pixels (not using depth mask) for color loss.
                # Therefore, for pixels with non-zero depth value, we sample near the surface,
                # since it is not a good idea to sample 16 points near (half even behind) camera,
                # for pixels with zero depth value, we sample uniformly from camera to max_depth.
                # 深度为0的地方就在相机和最大深度之间均匀采样，深度不为0的地方就在表面附近采样
                gt_none_zero_mask = gt_depth > 0
                gt_none_zero = gt_depth[gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(
                    0., 1., steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95*gt_depth_surface * \
                    (1.-t_vals_surface) + 1.05 * \
                    gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(
                    gt_depth.shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask,
                               :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(gt_depth)
                z_vals_surface_depth_zero = near_surface * \
                    (1.-t_vals_surface) + far_surface * (t_vals_surface)
                z_vals_surface_depth_zero.unsqueeze(
                    0).repeat((~gt_none_zero_mask).sum(), 1)
                z_vals_surface[~gt_none_zero_mask,
                               :] = z_vals_surface_depth_zero

        t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(
                torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]
        
        pointsf = pts.reshape(-1, 3)

        raw = self.eval_points(pointsf, decoders, c, stage, device)
        raw = raw.reshape(N_rays, N_samples+N_surface, -1)

        depth, uncertainty, color, weights = raw2outputs_nerf_color(
            raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
        if N_importance > 0:
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(self.perturb == 0.), device=device)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + \
                rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples+N_importance+N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, z_vals, rays_d, occupancy=self.occupancy, device=device)
            return depth, uncertainty, color, depth

        return depth, uncertainty, color, depth

        
    def render_img_close(self, c, decoders, c2w, device, stage, gt_depth = None):
            
            with torch.no_grad():
                
                H, W = self.H, self.W
                rays_o, rays_d = get_rays(
                    H,
                    W,
                    self.fx, 
                    self.fy,
                    self.cx,
                    self.cy,
                    c2w,
                    device
                )
                
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                
                depth_list = []
                uncertainty_list = []
                color_list = []
                semantic_list = []
                
                ray_batch_size = self.ray_batch_size
                gt_depth = gt_depth.reshape(-1)
                
                for i in range(0, rays_d.shape[0], ray_batch_size):
                    rays_d_batch = rays_d[i:i+ray_batch_size]
                    rays_o_batch = rays_o[i:i+ray_batch_size]
                    if gt_depth is None : 
                        ret = self.render_batch_ray(
                            c, 
                            decoders,
                            rays_d_batch,
                            rays_o_batch,
                            device,
                            stage,
                            gt_depth=None
                        )
                    else :
                        gt_depth_batch = gt_depth[i:i+ray_batch_size]
                        ret = self.render_batch_ray(
                            c, 
                            decoders,
                            rays_d_batch,
                            rays_o_batch,
                            device,
                            stage,
                            gt_depth=gt_depth_batch
                        )
                    
                    depth, uncertainty, color, semantic = ret
                    depth_list.append(depth.double())
                    uncertainty_list.append(uncertainty.double())
                    color_list.append(color)
                    semantic_list.append(semantic)
                    
                depth = torch.cat(depth_list, dim = 0).reshape(H, W)
                uncertainty = torch.cat(uncertainty_list, dim = 0).reshape(H, W)
                color = torch.cat(color_list, dim = 0).reshape(H, W, 3)
                semantic = torch.cat(semantic_list, dim = 0).reshape(H, W, -1)
                
                logits_2_label = lambda x : torch.argmax(torch.nn.functional.softmax(x, dim = -1), dim = -1)
                semantic = logits_2_label(semantic)
                
                return depth, uncertainty, color, semantic
            
            
    def render_img(self, c, decoders, c2w, device, stage, gt_depth = None):
            
            with torch.no_grad():
                
                H, W = self.H, self.W
                rays_o, rays_d = get_rays(
                    H,
                    W,
                    self.fx, 
                    self.fy,
                    self.cx,
                    self.cy,
                    c2w,
                    device
                )
                
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                
                depth_list = []
                uncertainty_list = []
                color_list = []
                semantic_list = []
                
                ray_batch_size = self.ray_batch_size
                gt_depth = gt_depth.reshape(-1)
                
                for i in range(0, rays_d.shape[0], ray_batch_size):
                    rays_d_batch = rays_d[i:i+ray_batch_size]
                    rays_o_batch = rays_o[i:i+ray_batch_size]
                    if gt_depth is None : 
                        ret = self.render_batch_ray(
                            c, 
                            decoders,
                            rays_d_batch,
                            rays_o_batch,
                            device,
                            stage,
                            gt_depth=None
                        )
                    else :
                        gt_depth_batch = gt_depth[i:i+ray_batch_size]
                        ret = self.render_batch_ray(
                            c, 
                            decoders,
                            rays_d_batch,
                            rays_o_batch,
                            device,
                            stage,
                            gt_depth=gt_depth_batch
                        )
                    
                    depth, uncertainty, color, semantic = ret
                    depth_list.append(depth.double())
                    uncertainty_list.append(uncertainty.double())
                    color_list.append(color)
                    semantic_list.append(semantic)
                    
                depth = torch.cat(depth_list, dim = 0).reshape(H, W)
                uncertainty = torch.cat(uncertainty_list, dim = 0).reshape(H, W)
                color = torch.cat(color_list, dim = 0).reshape(H, W, 3)
                semantic = torch.cat(semantic_list, dim = 0).reshape(H, W, -1)
                # semantic is a map consist of text probs of pixels
                 
                return depth, uncertainty, color, semantic
            

                
                
            
                    
                        
                    
                
                
            
            
            
            
             
            