from curses.ascii import DEL
from operator import imod
import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
from torch.nn import functional as F
import math
RAD=cfg.target_rad
DELTA = cfg.target_delta
# Reference car size in (length, height, width)
# for (car, cyclist, pedestrian) ('car','rider','person')
DIMENSION_REFERENCE = ((3.88, 1.63, 1.53),
                        (1.78, 1.70, 0.58),
                        (0.88, 1.73, 0.67))

def decode_dimension(cls_id, dim_offset, dim_ref):

    N_pre = cls_id.shape[0]

    dim_ref = torch.tensor(dim_ref).unsqueeze(0).repeat(N_pre, 1, 1)
    
    idx = torch.arange(N_pre)
    dim_select = dim_ref[idx, cls_id.long()]
    
    dimensions = dim_offset.exp() * dim_select.to(dim_offset)

    return dimensions

def decode_orientation(cam_loc, orient):

    rays = torch.atan(cam_loc[:,0] / (cam_loc[:,2] + 1e-7))
    alpha = torch.atan(orient[:,0] / (orient[:,1] + 1e-7))  # N_pre
    rot_y = alpha + rays
    
    rot_y[rot_y > torch.pi] -= 2.0 * torch.pi
    rot_y[rot_y < -1. * torch.pi] += 2.0 * torch.pi

    return rot_y


def world2cam(locations, pose):
    
    N_pre = locations.shape[0]
    T = pose[0,:3,3]
    R = pose[0,:3, :3]
    R_inv = torch.inverse(R).unsqueeze(0).repeat(N_pre,1,1)                     # torch.Size([N_pre, 3, 3])
    cam_loc = torch.matmul(R_inv, (locations - T).unsqueeze(-1)).squeeze(-1)    # torch.Size([N_pre, 3, 3]) @ torch.size(N_pre, 3, 1)

    return cam_loc

def rad_to_matrix(rot_y):
    
    N = rot_y.shape[0]
    cos, sin = rot_y.cos(), rot_y.sin()
    i_temp = torch.tensor([ [1, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 1]]).to(rot_y)
                            
    ry = i_temp.repeat(N, 1).view(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry

def cam2world(locations, pose):
    T = pose[0,:3,3].to(locations)
    R = pose[0,:3, :3].to(locations)
    N_pre = locations.shape[0]

    locations = torch.matmul(R.unsqueeze(0).repeat(N_pre, 1, 1), locations).transpose(1,2) + T  # torch.Size([N_pre, 3, 3]) @ torch.size(N_pre, 3, 1)
    
    return locations

def encode_3d_box(rotys, dims, locs, pose):
    '''
    input:
        rotys: [N], N denotes number of the boxes
        dims: [N, 3] in the order of l,h,w
        locs: [N, 3] center location xyz in camera coodinates
        pose: [1, 4, 4] 1 is batch size, now support only 1, 4*4 is [R,t]
    '''
    N = locs.shape[0]

    ry = rad_to_matrix(rotys).to(dims)

    dims = dims.view(-1, 1).repeat(1, 8)    # [N,3]→[N*3,8] 每3列代表1个框，每行代表h的8重复值
    dims[::3, :4], dims[1::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[1::3, :4], 0.5 * dims[2::3, :4]   # l,h,w 正轴
    dims[::3, 4:], dims[1::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[1::3, 4:], -0.5 * dims[2::3, 4:] # l,h,w 负轴

    index = torch.tensor([  [0, 1, 2, 3, 4, 5, 6, 7],
                            [4, 0, 5, 1, 2, 6, 3, 7],
                            [0, 1, 4, 5, 2, 3, 6, 7]]).repeat(N, 1).to(device = dims.device)  # [N*3,8]
    box_3d_object = torch.gather(dims, 1, index)    # [N*3,8]
    box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1)).permute(0,2,1) # [N,8,3]

    box_3d += locs.unsqueeze(1).repeat(1, 8, 1)     # [N,8,3]

    T = pose[0,:3,3].to(box_3d)
    R = pose[0,:3, :3].to(box_3d)
    box_3d_w = torch.matmul(R.unsqueeze(0).repeat(N, 1, 1), box_3d.permute(0,2,1)).permute(0,2,1) + T

    return box_3d_w, box_3d

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()   # 正例样本  torch.Size([B, N_pts, N_cls])
        negative_index = target.lt(1).float()   # 负例样本  torch.Size([B, N_pts, N_cls])

        negative_weights = torch.pow(1 - target, self.beta) # negative_weights
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.depth_crit = nn.HuberLoss(reduction='mean')
        self.weights_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.epsilon_max = 1.0
        self.epsilon_min = 0.2
        self.decay_speed = 0.00005

        self.regression_crit = nn.L1Loss(reduction='mean')
        self.cls_crit = FocalLoss()
    
    def get_gaussian(self, depth_gt, depth_samples):
        return torch.exp(-(depth_gt - depth_samples)**2 / (2*self.epsilon**2))

    def get_weights_gt(self, depth_gt, depth_samples):
        # near
        depth_gt = depth_gt.view(*depth_gt.shape, 1)
        weights = self.get_gaussian(depth_gt, depth_samples).detach()
        # empty and dist
        weights[torch.abs(depth_samples-depth_gt)>self.epsilon]=0
        # normalize
        weights = weights / torch.sum(weights,dim=2,keepdims=True).clamp(min=1e-6)
        return weights.detach()

    def kl_loss(self, weights_gt, weights_es):
        return torch.log(weights_gt * weights_es).sum()

    def forward(self, batch):
        output = self.net(batch)
        scalar_stats = {}
        loss = 0
        merge_list_car = [27, 28, 29, 30, 31]
        merge_list_box = [39]
        merge_list_park = [9]
        merge_list_gate = [35]
        depth_object = cfg.depth_object
        
        # rgb loss
        if 'rgb_0' in output.keys():
            color_loss = cfg.train.weight_color * self.color_crit(batch['rays_rgb'], output['rgb_0'])   # torch.Size([1, 2048, 3]) * torch.Size([1, 2048, 3])
            scalar_stats.update({'color_mse_0': color_loss})
            loss += color_loss
            psnr = -10. * torch.log(color_loss.detach()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({'psnr_0': psnr})
        
        # depth loss
        # if ('depth_0' in output.keys()) and ('depth' in batch) and cfg.use_depth == True:
        #     device = output['rgb_0'].device
        #     pred_depth = output['depth_0']
        #     gt_depth = batch['depth']
        #     semantic_filter = output['semantic_filter']
        #     semantic_filter = semantic_filter[..., 3]   # torch.Size([1, 2048, 10])
        #     mask_filter_depth = torch.zeros_like(gt_depth).to(semantic_filter) > 1
        #     for id in depth_object:
        #         mask_filter, _ = (semantic_filter == id).max(-1)
        #         mask_filter_depth = mask_filter_depth | mask_filter
        #     mask = (gt_depth>0) & (gt_depth<100) & mask_filter_depth
        #     if torch.sum(mask) < 0.5:
        #         depth_loss = torch.tensor(0.).to(device)
        #     else:
        #         depth_loss = self.depth_crit(gt_depth[mask], pred_depth[mask])
        #         depth_loss = depth_loss.clamp(max=0.1)
        #     scalar_stats.update({'depth_loss': depth_loss})
        #     loss += cfg.lambda_depth * depth_loss

        # # semantic_loss
        # if 'semantic_map_0' in output.keys():
        #     semantic_loss = 0.
        #     decay = 1.
        #     device = output['rgb_0'].device
        #     pseudo_label = batch['pseudo_label']

        #     # merge and filter 2d pseudo semantic
        #     for i in merge_list_car:
        #         pseudo_label[pseudo_label == i] = 26    # 取伪标签=i的下标，将其置为26
        #     for i in merge_list_box:
        #         pseudo_label[pseudo_label == i] = 41
        #     for i in merge_list_park:
        #         pseudo_label[pseudo_label == i] = 8
        #     for i in merge_list_gate:
        #         pseudo_label[pseudo_label == i] = 13
        #     if cfg.pseudo_filter == True:
        #         B, N_point, channel = output['semantic_map_0'].shape
        #         semantic_filter = output['semantic_filter']
        #         semantic_filter = semantic_filter[..., 3]
        #         for i in merge_list_car:
        #             semantic_filter[semantic_filter == i] = 26.
        #         for i in merge_list_box:
        #             semantic_filter[semantic_filter == i] = 41.
        #         for i in merge_list_park:
        #             semantic_filter[semantic_filter == i] = 8.
        #         for i in merge_list_gate:
        #             semantic_filter[semantic_filter == i] = 13.
        #         pseudo_label_temp = pseudo_label[..., None].repeat(1,1,semantic_filter.shape[-1])
        #         mask_filter, _ = (semantic_filter == pseudo_label_temp).max(-1)
        #         mask_filter = mask_filter[0]
        #         mask_sky = (pseudo_label == 23)
        #         mask_filter = (mask_sky | mask_filter).reshape(-1)
        #     else:
        #         mask_filter = torch.ones_like(pseudo_label.reshape(-1).long()).to(pseudo_label)>0

        #     cross_entropy = nn.CrossEntropyLoss()
        #     nll = nn.NLLLoss()
        #     # 2d pred
        #     B, N_point, channel = output['semantic_map_0'].shape
        #     if torch.sum(mask_filter) != 0: # 00摄像头的标签不为0，计算loss，01的标签全为0，loss为0
        #         semantic_loss_2d_pred = nll(torch.log(output['semantic_map_0'].reshape(-1 ,channel)[mask_filter]+1e-5), pseudo_label.reshape(-1).long()[mask_filter])
        #     else:
        #         semantic_loss_2d_pred = torch.tensor(0.).to(device)
        #     semantic_loss_2d_pred = decay * cfg.lambda_semantic_2d  * semantic_loss_2d_pred
        #     semantic_loss += semantic_loss_2d_pred
            
        #     # 2d fix
        #     semantic_loss_2d_fix = nll(torch.log(output['fix_semantic_map_0'].reshape(-1 ,channel)+1e-5), pseudo_label.reshape(-1).long())
        #     semantic_loss_2d_fix = cfg.lambda_fix * semantic_loss_2d_fix
        #     semantic_loss += semantic_loss_2d_fix

        #     # 3d primitive
        #     semantic_gt = output['semantic_bbox_gt']
        #     idx0_bg, idx1_bg, idx2_bg = torch.where(semantic_gt==-1.)
        #     inf = torch.empty_like(semantic_gt).fill_(-float('inf'))
        #     semantic_gt = torch.where(semantic_gt == 0., inf, semantic_gt)
        #     m = nn.Softmax(dim=2)
        #     semantic_gt = m(semantic_gt).to(device)
        #     semantic_gt[idx0_bg, idx1_bg, idx2_bg] = 0.
        #     msk_max, _ = semantic_gt.reshape(-1 ,channel).max(1)
        #     msk = (msk_max >= 0.99999) & (output['weights_0'].reshape(-1) > cfg.weight_th)
        #     if torch.sum(msk).item() != 0:
        #         semantic_loss_3d = cross_entropy(output['points_semantic_0'].reshape(-1 ,channel)[msk, :], semantic_gt.reshape(-1 ,channel)[msk, :])
        #     else:
        #         semantic_loss_3d = torch.tensor(0.).to(device)
        #     semantic_loss_3d = cfg.lambda_3d * semantic_loss_3d
        #     semantic_loss += semantic_loss_3d
            
        #     if (cfg.use_pspnet == True) and (batch['stereo_num'] == 1): # 01摄像头的loss为0
        #         semantic_loss = torch.tensor(0.).to(device)
        #         semantic_loss_3d = torch.tensor(0.).to(device)
        #         semantic_loss_2d_pred = torch.tensor(0.).to(device)
        #         semantic_loss_2d_fix = torch.tensor(0.).to(device)
        #     scalar_stats.update({'semantic_loss_2d_pred': semantic_loss_2d_pred})
        #     scalar_stats.update({'semantic_loss_2d_fix': semantic_loss_2d_fix})
        #     scalar_stats.update({'semantic_loss_3d': semantic_loss_3d})
        #     scalar_stats.update({'semantic_loss': semantic_loss})
        #     loss += cfg.semantic_weight * semantic_loss

        if batch['stereo_num'] == 0 and 'box3d_cls_prob_0' in output.keys():
            pre_cls_prob = output['box3d_cls_prob_0']
            pre_dimension_offset= output['box3d_dimension_0']
            pre_orientation = output['box3d_local_angle_0']
            xyz = output['xyz']

            box3d_label = batch['box3d_label']
            box3d_rotys = batch['box3d_rotys']
            box3d_ct_loc = batch['box3d_ct_loc']
            box3d_hwl = batch['box3d_hwl']
            box3d_reg_tgt = batch['box3d_reg_tgt']

            B, N_rays, N_samples, _ = xyz.shape
            N_gt_3dboxes = box3d_label.shape[-1]
            N_pts = N_rays*N_samples
            N_cls = pre_cls_prob.shape[-1]

            pre_dimension_offset = pre_dimension_offset.reshape(B, N_pts, -1)   # B, N_pts, 3
            pre_orientation = pre_orientation.reshape(B, N_pts, -1)  # B, N_pts, 2
            pre_cls_prob = pre_cls_prob.reshape(B, N_pts, -1)   # B, N_pts, N_cls

            # for classification
            cls_tgt = torch.zeros_like(pre_cls_prob)   # B, N_pts, N_cls
            for j in range(B):
                for i in range(N_cls):
                    cls_idx = box3d_label[j].eq(i)
                    cls_ct_loc = box3d_ct_loc[j][cls_idx]  # N_cls_pts, 3

                    bias = torch.pow(xyz[j].reshape(N_pts, 3).unsqueeze(1) - cls_ct_loc.unsqueeze(0),2).sum(dim=-1)  # 计算采样点与gt 3dbox 中心的距离
                    bias = torch.sqrt(bias)
                    
                    gt_ct_loc_idx = bias.lt(DELTA).sum(dim=-1).eq(1)    # 在gt中心点领域内的，不在是~gt_ct_loc_idx
                    cls_tgt[j,gt_ct_loc_idx, i] = 1.

                    gt_near_loc_idx = bias.lt(RAD).sum(dim=-1).eq(1)          # 小于半径的，且仅在一个球内的
                    gt_near_loc_idx = gt_near_loc_idx & ~gt_ct_loc_idx        # 小于半径且大于等于邻域的  
                    cls_tgt[j, gt_near_loc_idx, i] += bias[gt_near_loc_idx][bias[gt_near_loc_idx].lt(RAD)]/(RAD-DELTA)
                
            cls_loss = self.cls_crit(pre_cls_prob, cls_tgt)
            
            # for regression
            # 计算每一个采样与所有的gt3d框中心的距离
            centers = box3d_ct_loc.unsqueeze(1)
            dist = torch.pow(xyz.reshape(B, N_pts, -1).unsqueeze(2) - centers,2).sum(dim=-1)    # torch.Size([B, N_pts, N_gt_3dboxes])
            dist = torch.sqrt(dist)
            pt_index = dist <= RAD  # 点与中心距离小于半径的为球体内点  B, N_pts, N_gt_3dboxes

            edge_pts_mask = pt_index.sum(dim=-1)>1  # 如果一个点同时落入两个球体，视为是边缘容易冲突点，该点上不预测回归值
            pt_index[edge_pts_mask] = False

            # 取所有位于球体中的点，loss其预测回归量
            in_pt_indexs = pt_index.sum(dim=-1)==1  # 所有位于球体中的点的index B,N_gts
            N_in_pt=in_pt_indexs.sum(dim=-1).item()
            
            reg_loss = torch.tensor(0.0)
            if N_in_pt != 0:
                
                # 取对应的预测几何量
                in_pt_pre_dim_offset = pre_dimension_offset[in_pt_indexs]       # torch.Size([N_in_pt, 3])
                in_pt_pre_orient = pre_orientation[in_pt_indexs]   # torch.Size([N_in_pt, 2])
                
                # box_index是每个落入球体的点对应的3dbox
                pt_index = pt_index.view(-1, N_gt_3dboxes)  # B,N_pts,N_gt_3dboxes → B*N_pts, N_gt_3dboxes
                idx0, box_index = torch.where(pt_index==True)
                # 取对应的gt值
                gt_label = box3d_label.unsqueeze(1).repeat(1,N_pts,1).view(B*N_pts, N_gt_3dboxes)[idx0, box_index]          # torch.Size([N_in_pt])
                gt_rotys = box3d_rotys.unsqueeze(1).repeat(1,N_pts,1).view(B*N_pts, N_gt_3dboxes)[idx0, box_index]          # torch.Size([N_in_pt]
                gt_ct_loc = box3d_ct_loc.unsqueeze(1).repeat(1,N_pts,1,1).view(B*N_pts, N_gt_3dboxes, 3)[idx0, box_index]   # torch.Size([N_in_pt, 3])
                gt_hwl = box3d_hwl.unsqueeze(1).repeat(1,N_pts,1,1).view(B*N_pts, N_gt_3dboxes, 3)[idx0, box_index]         # torch.Size([N_in_pt, 3])
                gt_reg_tgt = box3d_reg_tgt.unsqueeze(1).repeat(1,N_pts,1,1,1).view(B*N_pts, N_gt_3dboxes, 8, 3)[idx0, box_index]    # torch.Size([N_in_pt, 8, 3])

                in_pt_pre_dimension = decode_dimension(gt_label, in_pt_pre_dim_offset, DIMENSION_REFERENCE) # torch.Size([N_in_pt, 3]), l,h,w

                cam_ct_loc = world2cam(gt_ct_loc, batch['pose'])
                in_pt_pre_rot_y = decode_orientation(cam_ct_loc, in_pt_pre_orient)  # torch.Size([N_in_pt]

                ## for debug
                # T = batch['pose'][0,:3,3].to(cam_ct_loc)
                # R = batch['pose'][0,:3, :3].to(cam_ct_loc)
                ## cam_center_loc
                # N_pre = cam_ct_loc.shape[0]
                # locations = torch.matmul(R.unsqueeze(0).repeat(N_pre, 1, 1), cam_ct_loc.unsqueeze(-1)).squeeze(-1) + T 
                ## cam_corners
                # N = gt_reg_tgt.shape[0]
                # gt_reg_tgt[...,None]
                # cam_corners = torch.matmul(torch.inverse(R)[None, None].repeat(N,8,1,1),(gt_reg_tgt- T)[...,None]).squeeze(-1)

                # rotys = in_pt_pre_rot_y
                # dims = in_pt_pre_dimension
                # locs = gt_ct_loc

                gt_lhw = gt_hwl[:,[2,0,1]]
                encode_gt_reg_w, encode_gt_reg_camara  = encode_3d_box(gt_rotys, gt_lhw, cam_ct_loc, batch['pose'])
                pred_3dbox_dims_w, pred_3dbox_dims_camera = encode_3d_box(gt_rotys, in_pt_pre_dimension, cam_ct_loc, batch['pose'])
                pred_3dbox_rotys_w, pred_3dbox_rotys_camera = encode_3d_box(in_pt_pre_rot_y, gt_lhw, cam_ct_loc, batch['pose'])

                reg_loss_dim = self.regression_crit(pred_3dbox_dims_w, encode_gt_reg_w)
                reg_loss_ori = self.regression_crit(pred_3dbox_rotys_w, encode_gt_reg_w)
                reg_loss = reg_loss_dim + reg_loss_ori

            detection_loss = cfg.detection_weight * (cls_loss + reg_loss)
            loss += detection_loss
            scalar_stats.update({'cls_loss': cls_loss})
            scalar_stats.update({'reg_loss': reg_loss})
            scalar_stats.update({'detection_loss': detection_loss})
        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

