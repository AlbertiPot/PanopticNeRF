import numpy as np
import cv2
import random
from lib.config import cfg
from torch import nn
import torch
from imgaug import augmenters as iaa
import collections

def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array([0.1, 0.1, 0.1, 1.])
    hwf = c2w[3:, :]
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # import ipdb; ipdb.set_trace()
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 0))
    return render_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr

def convert_id_instance(intersection):
    instance2id = {}
    id2instance = {}
    instances = np.unique(intersection[..., 2])
    for index, inst in enumerate(instances):
        instance2id[index] = inst
        id2instance[inst] = index
    semantic2instance = collections.defaultdict(list)
    semantics = np.unique(intersection[..., 3])
    for index, semantic in enumerate(semantics):
        if semantic == -1:
            continue
        semantic_mask = (intersection[..., 3] == semantic)
        instance_list = np.unique(intersection[semantic_mask, 2])
        for inst in  instance_list:
            semantic2instance[semantic].append(id2instance[inst])
    instances = np.unique(intersection[..., 2])
    instance2semantic = {}
    for index, inst in enumerate(instances):
        if inst == -1:
            continue
        inst_mask = (intersection[..., 2] == inst)
        semantic = np.unique(intersection[inst_mask, 3])
        instance2semantic[id2instance[inst]] = semantic
    instance2semantic[id2instance[-1]] = 23
    return instance2id, id2instance, semantic2instance, instance2semantic

def to_cuda(batch, device=torch.device('cuda:'+str(cfg.local_rank))):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        batch = batch.to(device)
    return batch

def build_rays(ixt, c2w, H, W):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)  # [0,0,1], [1,0,1]，[201]先沿着W排列，再沿着H排列, Z轴是1
    XYZ = XYZ @ np.linalg.inv(ixt[:3, :3]).T    # 新的XYZ是车辆坐标系的行向量 ix 内参,3*3矩阵 XYZ是188*704*3，取最后一维度是行向量 [0, 0, 1]，所以需要转置： K矩阵乘{车辆坐标系向量} = 图像坐标系列向量
    XYZ = XYZ @ c2w[:3, :3].T # 新XYZ是世界坐标系的行向量， X世界.T = Xcam.T @ R.T
    
    # 图像像素点在世界坐标系的位置为该像素点对应的方向向量
    # 相机外参的平移向量t为ray的光线原点
    rays_d = XYZ.reshape(-1, 3) # [188,704,3] → [188*704,3] 共132352个行向量 位置， 
    rays_o = c2w[:3, 3] #取外参最后一列的平移向量t（行向量）为坐标原点，相机在世界坐标系中相对于原点的偏移
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1)   # 每一行是 ray_o的3个坐标+rayd的3个坐标，前三个坐标都是一样的，是该张图片对应相机的坐标