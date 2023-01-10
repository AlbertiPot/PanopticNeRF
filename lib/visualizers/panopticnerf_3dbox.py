import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from lib.config import cfg
import os
from tools.kitti360scripts.helpers.labels import id2label, labels
import torch.nn as nn
from torch.functional import norm

from shapely.geometry import Polygon

from matplotlib import pyplot as plt

from lib.train.losses.panopticnerf import decode_dimension, decode_orientation, encode_3d_box, DIMENSION_REFERENCE

def assigncolor(globalids, gttype='semantic'):
    if not isinstance(globalids, (np.ndarray, np.generic)):
        globalids = np.array(globalids)[None]
    color = np.zeros((globalids.size, 3))
    # semanticid = globalids
    for uid in np.unique(globalids):
        # semanticid, instanceid = global2local(uid)
        if gttype == 'semantic':
            try:
                color[globalids == uid] = id2label[uid].color
            except:
                color[globalids == uid] = (0, 0, 0)  # stuff objects in instance mode
                print("warning! unkown category!")
        else:
            color[globalids == uid] = (96, 96, 96)  # stuff objects in instance mode
    color = color.astype(np.float) / 255.0
    return color

def nms_pt(heat_map, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool1d(heat_map,
                        kernel_size=kernel,
                        stride=1,
                        padding=pad)
    eq_index = (hmax == heat_map).float()

    return heat_map * eq_index

def select_topk(heat_map, K=20):
    '''
    Args:
        heat_map: heat_map in [N_cls, N_pts]
        K: top k samples to be selected
        score: detection threshold

    Returns:
    '''
    N_cls, N_pts = heat_map.shape

    # Select topK examples in each channel
    topk_scores_all, topk_inds_all = torch.topk(heat_map, K)    # N_cls, K

    # Select topK examples across channel
    topk_scores_all = topk_scores_all.view(-1)   # N_cls, K → N_cls*K
    topk_scores, topk_inds = torch.topk(topk_scores_all, K) # K
    topk_indices = torch.gather(topk_inds_all.view(-1),dim=0, index=topk_inds)

    top_cls = torch.div(topk_inds, K, rounding_mode='trunc').long()# topk_inds 是下标 = cls_id × K，除以K还原到cls_id # K

    return topk_scores, top_cls, topk_indices   # K


def plot_3dbox(img, u, v, color):
    '''
    u,v: [N_box,8]
    '''
    lines = [(0,2),[1,3],(5,7),(4,6),(2,7),(0,5),(1,4),(3,6),(0,1),(2,3),(5,4),(7,6)]
    
    cv_img = img.copy()
    
    for box_u,box_v in zip(u,v):
        box_u = box_u.numpy()
        box_v = box_v.numpy()
        for line in lines:
            (pt1 ,pt2) = line
            cv2.line(cv_img, (box_u[pt1], box_v[pt1]), (box_u[pt2], box_v[pt2]), color,1)

    return cv_img

def cam2image(cam_cood, K):
    '''
    K: [3, 3]
    cam_cood: [N_box, 8, 3]
    '''
    K = K.to(cam_cood)
    projected_pts = torch.matmul(K[None,None], cam_cood[..., None]).squeeze(-1)
    
    z = projected_pts[:,:, 2]
    z[z==0] = 1e-7
    u = torch.div(projected_pts[:,:,0], z, rounding_mode='trunc').long()
    v = torch.div(projected_pts[:,:,1], z, rounding_mode='trunc').long()

    return u,v,z


def box_iou_3d(corner1, corner2):
    top_face_inds = [1,3,6,4]   # must in this order
    bottom_face_inds = [0,2,7,5]
    # for height overlap, since y face down, use the negative y
    min_h_a = corner1[bottom_face_inds, 1].sum() / 4.0
    max_h_a = corner1[top_face_inds, 1].sum() / 4.0
    min_h_b = corner2[bottom_face_inds, 1].sum() / 4.0
    max_h_b = corner2[top_face_inds, 1].sum() / 4.0

	# overlap in height
    h_max_of_min = max(min_h_a, min_h_b)
    h_min_of_max = min(max_h_a, max_h_b)
    h_overlap = max(0, h_min_of_max - h_max_of_min)

    if h_overlap == 0:
        return 0

    # x-z plane overlap
    box1, box2 = corner1[top_face_inds][:,[0,2]].cpu().numpy(), corner2[top_face_inds][:,[0,2]].cpu().numpy(),
    bottom_a, bottom_b = Polygon(box1), Polygon(box2)
    if bottom_a.is_valid and bottom_b.is_valid:
        # check is valid, A valid Polygon may not possess any overlapping exterior or interior rings.
        bottom_overlap = bottom_a.intersection(bottom_b).area

    overlap_3d = bottom_overlap * h_overlap 
    union3d = bottom_a.area * (max_h_a - min_h_a) + bottom_b.area * (max_h_b - min_h_b) - overlap_3d

    return overlap_3d / union3d

def nms_3d(boxes, scores, classes, thresh=0.5):
    """3D NMS for rotated boxes.

    Args:
        boxes (torch.Tensor): 8 corners of each box, [N, 8, 3].
        scores (torch.Tensor): Scores of each box, [N].
        classes (torch.Tensor): Class of each box [N].
        thresh (float): IoU threshold for nms.

    Returns:
        torch.Tensor: Indices of selected boxes.
    """
    score_sorted = torch.argsort(scores)    # ascending
    pick = []
    while (score_sorted.shape[0] != 0):
        last = score_sorted.shape[0]
        i = score_sorted[-1]
        pick.append(i)
        iou_list = []
        for ind in score_sorted[:last - 1]:
            iou_list.append(box_iou_3d(boxes[i], boxes[ind]))
        iou = torch.tensor(iou_list).to(boxes)
        classes1 = classes[i]
        classes2 = classes[score_sorted[:last - 1]]
        
        iou = iou * (classes1 == classes2).float()
        score_sorted = score_sorted[torch.nonzero(iou <= thresh, as_tuple=False).flatten()]

    indices = torch.tensor(pick, dtype=torch.long)
    
    return indices

class Visualizer:
    def __init__(self, ):
        self.color_crit = lambda x, y: ((x - y)**2).mean()
        self.mse2psnr = lambda x: -10. * np.log(x) / np.log(torch.tensor([10.]))
        self.psnr = []

    def visualize(self, output, batch):
        b = len(batch['rays'])  # torch.Size([B, H*W, rays_o+rays_d])
        for b in range(b):
            
            result_dir = cfg.result_dir
            result_dir = os.path.join(result_dir, batch['meta']['sequence'][0])
            print(result_dir)
            os.system("mkdir -p {}".format(result_dir))

            h, w = batch['meta']['h'][b].item(), batch['meta']['w'][b].item()      
            img_id = int(batch["meta"]["tar_idx"].item())
            gt_img = batch['rays_rgb'][b].reshape(h, w, 3).detach().cpu().numpy()   # (188, 704, 3) H,W,3
            cv2.imwrite('{}/img{:04d}_gt.png'.format(result_dir, img_id), cv2.cvtColor(gt_img*255, cv2.COLOR_RGB2BGR))
            
            pred_img = torch.clamp(output['rgb_0'][b], min=0.,max=1.).reshape(h, w, 3).detach().cpu().numpy()   # colored img
            cv2.imwrite('{}/img{:04d}_novelview.png'.format(result_dir, img_id), cv2.cvtColor(pred_img*255, cv2.COLOR_RGB2BGR))

            pre_cls_prob = output['box3d_cls_prob_0'][b]
            pre_dimension_offset= output['box3d_dimension_0'][b]
            pre_orientation = output['box3d_local_angle_0'][b]
            xyz = output['xyz'][b]

            N_rays, N_samples, N_cls = xyz.shape
            N_pts = N_rays*N_samples
            
            # handle prediction 3d boxes
            pre_cls_prob = pre_cls_prob.reshape(N_pts, -1)   # N_pts, N_cls

            # nms 针对某一类，筛选领域内最大的点
            heatmap = nms_pt(pre_cls_prob.transpose(0,1))   # 沿着点方向    # shape: N_cls, N_pts
            # heatmap = pre_cls_prob.transpose(0,1)
            K = cfg.max_detection
            topk_scores, top_cls, topk_indices = select_topk(heatmap, K=K)

            pre_dimension_offset = pre_dimension_offset.view(-1, 3)[topk_indices]   # K,3
            pre_orientation = pre_orientation.view(-1,2)[topk_indices]  # K, 2
            xyz = xyz.view(-1,3)[topk_indices]  # K,3

            T = batch['pose'][b][:3,3].to(xyz)
            R = batch['pose'][b][:3, :3].to(xyz)
            R_inv = torch.inverse(R).unsqueeze(0).repeat(K,1,1)
            pred_box_ct_cam = torch.matmul(R_inv, (xyz - T).unsqueeze(-1)).squeeze(-1)
            pred_3dbox_dims = decode_dimension(top_cls, pre_dimension_offset, DIMENSION_REFERENCE)
            prde_3dbox_rot_y = decode_orientation(pred_box_ct_cam, pre_orientation)

            pred_3dbox_w, pred_3dbox_cam = encode_3d_box(prde_3dbox_rot_y, pred_3dbox_dims, pred_box_ct_cam, batch['pose'])
            
            # handel ground-truth 3d boxes
            gt_lhw = batch['box3d_hwl'][b][:, [2,0,1]]
            gt_ct_world = batch['box3d_ct_loc'][b]
            R_inv = torch.inverse(R)[None]
            gt_ct_cam = torch.matmul(R_inv.to(gt_ct_world), (gt_ct_world - T).unsqueeze(-1)).squeeze(-1)
            gt_3dbox_w, gt_3dbox_c = encode_3d_box(batch['box3d_rotys'][b], gt_lhw, gt_ct_cam, batch['pose'])

            pred_box_inds = nms_3d(pred_3dbox_cam, topk_scores, top_cls, cfg.nms_thr)

            # plot 3d boxes
            # TODO:change to key of classes
            class_colors = {
                0:(0,255,0),
                1:(255,255,0),
                2:(0,255,255)
            }
            color = class_colors[0]

            u_gt, v_gt, z_gt = cam2image(gt_3dbox_c.cpu(), batch['intrinsics'][b].cpu())
            u_pred, v_pred, z_pred = cam2image(pred_3dbox_cam[pred_box_inds].detach().cpu(), batch['intrinsics'][b].cpu())

            gt_3dbox_img = plot_3dbox(gt_img*255, u_gt, v_gt, color)
            pred_3dbox_img = plot_3dbox(gt_img*255, u_pred, v_pred, color)
            cv2.imwrite('{}/img{:04d}_pred_3dbox.png'.format(result_dir, img_id), cv2.cvtColor(pred_3dbox_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite('{}/img{:04d}_gt_3dbox.png'.format(result_dir, img_id), cv2.cvtColor(gt_3dbox_img, cv2.COLOR_RGB2BGR))
            
            # plot top view
            fig = plt.figure(figsize=(3, 8))
            ax1 = plt.axes()
            top_face_inds = [1,3,6,4]
            for box in gt_3dbox_c:
                pts = box[top_face_inds][:, [0,2]].cpu().numpy()
                for i1,i2 in [(0,1),(1,2),(2,3),(3,0)]:
                    ax1.plot((pts[i1,0],pts[i2,0]),(pts[i1,1],pts[i2,1]), color='r')

            for box in pred_3dbox_cam[pred_box_inds]:
                pts = box[top_face_inds][:, [0,2]].cpu().numpy()
                for i1,i2 in [(0,1),(1,2),(2,3),(3,0)]:
                    ax1.plot((pts[i1,0],pts[i2,0]),(pts[i1,1],pts[i2,1]), color='b')
            fig.savefig('{}/img{:04d}_topview.png'.format(result_dir, img_id))
            
            # save results
            pred_3dbox_results= {'world' : pred_3dbox_w[pred_box_inds].detach().cpu(), 
                               'camera': pred_3dbox_cam[pred_box_inds].detach().cpu(),
                               'K' : batch['intrinsics'][b].cpu(),
                               'pose' : batch['pose'][b].cpu(),
                               'label_w' : gt_3dbox_w.cpu(),
                               'label_c' : gt_3dbox_c.cpu(),}
            torch.save(pred_3dbox_results, '{}/img{:04d}_pred_3dbox.pt'.format(result_dir, img_id))
            stop = 0           