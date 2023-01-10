from http.client import ImproperConnectionState
import imp
from tkinter import N
from matplotlib import cm
import numpy as np
import os
from glob import glob
from lib.utils.data_utils import *
from lib.config import cfg, args
import imageio
from multiprocessing import Pool
from tools.kitti360scripts.helpers.annotation import Annotation3D
from tools.kitti360scripts.helpers.annotation_3dbox import Annotation3D as Annotation3D_3dbox
from tools.kitti360scripts.helpers.annotation_3dbox import Annotation2DInstance, global2local, local2global
from tools.kitti360scripts.helpers.labels import labels, name2label, id2label, assureSingleInstanceName
from tools.kitti360scripts.helpers.project import CameraPerspective
import cv2
import copy
import torch
from lib.datasets.kitti360.anno_helpers import cal_hwl, get_yaw

f = lambda k, v : v[k]
_CLASS_IDX = {
    'car':0,
    'rider':1,
    'person':2
}

class Dataset:
    def __init__(self, cam2world_root, img_root, bbx_root, data_root, sequence, pseudo_root, scene, split, detect_classes):
        super(Dataset, self).__init__()
        # path and initialization
        self.split = split
        self.sequence = sequence
        self.start = cfg.start
        self.pseudo_root = pseudo_root
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.scene = scene
        # self.max_objects = max_objects
        # load image_ids
        train_ids = np.arange(self.start, self.start + cfg.train_frames)    # 训练图像的id
        # test_ids = np.arange(self.start, self.start + cfg.train_frames)
        test_ids = np.array(cfg.val_list)   # test图像的id
        if split == 'train':
            self.image_ids = np.array([i for i in train_ids if i not in test_ids])
        elif split == 'val':
            self.image_ids = test_ids

        # load intrinsics
        calib_dir = os.path.join(data_root, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')    # 相机内参
        self.load_intrinsic(self.intrinsic_file)
        self.H = int(self.height * cfg.ratio)
        self.W = int(self.width  * cfg.ratio)   # 图像下采样0.5倍
        self.K_00[:2] = self.K_00[:2] * cfg.ratio   # 3*4，第3行是0010，所以只用前两行乘以ratio
        self.K_01[:2] = self.K_01[:2] * cfg.ratio
        self.intrinsic_00 = self.K_00[:, :-1]   # 取左上角3*3矩阵
        self.intrinsic_01 = self.K_01[:, :-1]
 
        # load cam2world poses
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.pose_file = os.path.join(data_root, 'data_poses', sequence, 'poses.txt')   # each line [frame_idx, 3*4车辆坐标系到世界坐标系的刚体（平移+旋转）变换矩阵]
        poses = np.loadtxt(self.pose_file)  # 10514*13, 10514是全部帧的数量
        frames = poses[:, 0]    # frames idex
        poses = np.reshape(poses[:, 1:], [-1, 3, 4]) # 10514个 3*4外参矩阵
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt') # 相机到车辆坐标系
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']  # 1号相机到车辆坐标系的映射
        for line in open(cam2world_root, 'r').readlines():  
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)    # # 0号相机，所有帧从相机坐标系到世界坐标系的映射
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect)) # 1号相机，所有帧从相机坐标系到世界坐标系的映射 ，通过1号相机的内参等算得
        self.translation = np.array(cfg.center_pose)
        
        # load 3dbbox
        self.anno_3dbox = Annotation3D_3dbox(os.path.join(data_root, 'data_3d_bboxes'),sequence)
        # load camea for frame idx and poses
        self.camera_0 = CameraPerspective(root_dir=data_root, seq=sequence, cam_id=0)
        self.camera_1 = CameraPerspective(root_dir=data_root, seq=sequence, cam_id=1)
        self.camera_0.load_intrinsics(self.camera_0.intrinsic_file)
        self.camera_1.load_intrinsics(self.camera_1.intrinsic_file)

        self.camera_0.width = int(self.camera_0.width  * cfg.ratio)
        self.camera_0.height = int(self.camera_0.height  * cfg.ratio)
        self.camera_0.K[:2] = self.camera_0.K[:2] * cfg.ratio 

        self.camera_1.width = int(self.camera_1.width  * cfg.ratio)
        self.camera_1.height = int(self.camera_1.height  * cfg.ratio)
        self.camera_1.K[:2] = self.camera_1.K[:2] * cfg.ratio 

        label2DPath = os.path.join(data_root, 'data_2d_semantics', 'train')
        annotation2DInstance = Annotation2DInstance(os.path.join(label2DPath, sequence, 'image_00'))

        # load images and corresponding 3d bboxes
        self.visible_id = os.path.join(data_root, 'visible_id', sequence)
        self.images_list_00 = {}
        self.images_list_01 = {}
        self.images_objects_visi_globalId = {}
        self.box3d_list_00 = {}
        self.box3d_list_01 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            image_file_00 = os.path.join(img_root, 'image_00/data_rect/%s.png' % frame_name)
            image_file_01 = os.path.join(img_root, 'image_01/data_rect/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                raise RuntimeError('%s does not exist!' % image_file_00)
            self.images_list_00[idx] = image_file_00
            self.images_list_01[idx] = image_file_01
            
            # objects visiable globalId
            visi_globalId = []
            for k, v in annotation2DInstance.instanceDict.items():  # k是globalId，v是出现k对应obj的帧list， num.png
                if '%010d.png'%idx in v:
                    visi_globalId.append(k)
            self.images_objects_visi_globalId[idx] = visi_globalId
            
            # all 3d bboxes for this frame
            frame_annos_00 = []
            frame_annos_01 = []
            for k,v in self.anno_3dbox.objects.items():
                anno_0 = {}
                anno_1 = {}
                if len(v.keys())==1 and (-1 in v.keys()): # show static only
                    obj3d = v[-1]
                    obj3d_globalId = local2global(obj3d.semanticId, obj3d.instanceId)
                    class_name = id2label[obj3d.semanticId].name

                    if not class_name in detect_classes:
                            continue
                    if not obj3d_globalId in visi_globalId:
                        continue

                    self.camera_0(obj3d, int(idx))
                    obj3d_01 = copy.deepcopy(obj3d)
                    self.camera_1(obj3d_01, int(idx))

                    anno_0['globalId'] = obj3d_globalId
                    anno_1['globalId'] = obj3d_globalId

                    anno_0['class'] = class_name
                    anno_1['class'] = class_name

                    anno_0['label'] = _CLASS_IDX[class_name]
                    anno_1['label'] = _CLASS_IDX[class_name]

                    center_locations = obj3d.vertices[0]*0.5 + obj3d.vertices[6]*(1-0.5)
                    anno_0['locations'] = center_locations
                    anno_1['locations'] = center_locations

                    hwl = cal_hwl(obj3d)
                    anno_0['dimensions'] = hwl
                    anno_1['dimensions'] = hwl

                    regression = obj3d.vertices
                    anno_0['regression'] =regression # [8,3]
                    anno_1['regression'] =regression
                    
                    # for debug
                    # crpose = self.camera_0.cam2world[int(idx)]
                    # T = crpose[:3,  3]- self.translation
                    # R = crpose[:3, :3]
                    # anno_0['regression'] =self.camera_0.world2cam(np.array([regression-self.translation]),R,T,inverse=True)
                    # anno_0['cam_loc'] = self.camera_0.world2cam(np.asarray([center_locations-self.translation]),R,T,inverse=True)

                    yaw, alpha = get_yaw(obj3d, self.camera_0, int(idx))
                    anno_0['rot_y']  = yaw
                    anno_0['alpha'] = alpha
                    yaw_1, alpha_1 = get_yaw(obj3d_01, self.camera_1, int(idx))
                    anno_1['rot_y']  = yaw_1
                    anno_1['alpha'] = alpha_1
                    
                    frame_annos_00.append(anno_0)
                    frame_annos_01.append(anno_1)
                else:   # dynamic
                    if idx in v.keys():
                        obj3d = v[idx]
                        obj3d_globalId = local2global(obj3d.semanticId, obj3d.instanceId)
                        class_name = id2label[obj3d.semanticId].name

                        if not class_name in detect_classes:
                            continue
                        if not obj3d_globalId in visi_globalId:
                            continue

                        self.camera_0(obj3d, int(idx))
                        obj3d_01 = copy.deepcopy(obj3d)
                        self.camera_1(obj3d_01, int(idx))

                        anno_0['globalId'] = obj3d_globalId
                        anno_1['globalId'] = obj3d_globalId

                        anno_0['class'] = class_name
                        anno_1['class'] = class_name

                        anno_0['label'] = _CLASS_IDX[class_name]
                        anno_1['label'] = _CLASS_IDX[class_name]

                        center_locations = obj3d.vertices[0]*0.5 + obj3d.vertices[6]*(1-0.5)    # world_cood, same for two cameras
                        anno_0['locations'] = center_locations
                        anno_1['locations'] = center_locations

                        hwl = cal_hwl(obj3d)
                        anno_0['dimensions'] = hwl
                        anno_1['dimensions'] = hwl

                        regression = obj3d.vertices
                        anno_0['regression'] =regression # [8,3]
                        anno_1['regression'] =regression

                        yaw, alpha = get_yaw(obj3d, self.camera_0, int(idx))
                        anno_0['rot_y']  = yaw
                        anno_0['alpha'] = alpha
                        yaw_1, alpha_1 = get_yaw(obj3d_01, self.camera_1, int(idx))
                        anno_1['rot_y']  = yaw_1
                        anno_1['alpha'] = alpha_1
                        
                        frame_annos_00.append(anno_0)
                        frame_annos_01.append(anno_1)
            self.box3d_list_00[idx] = frame_annos_00
            self.box3d_list_01[idx] = frame_annos_01

        # load intersections
        self.bbx_intersection_root = os.path.join(data_root, 'bbx_intersection')
        self.intersections_dict_00 = {}
        self.intersections_dict_01 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            intersection_file_00 = os.path.join(self.bbx_intersection_root,self.sequence,str(idx) + '.npz')
            intersection_file_01 = os.path.join(self.bbx_intersection_root, self.sequence,str(idx) + '_01.npz')
            if not os.path.isfile(intersection_file_00):
                raise RuntimeError('%s does not exist!' % intersection_file_00)
            self.intersections_dict_00[idx] = intersection_file_00
            self.intersections_dict_01[idx] = intersection_file_01

        # load annotation3D
        self.annotation3D = Annotation3D(bbx_root, sequence)
        self.bbx_static = {}
        self.bbx_static_annotationId = []
        self.bbx_static_center = []
        for annotationId in self.annotation3D.objects.keys():
            if len(self.annotation3D.objects[annotationId].keys()) == 1:
                if -1 in self.annotation3D.objects[annotationId].keys():
                    self.bbx_static[annotationId] = self.annotation3D.objects[annotationId][-1]
                    self.bbx_static_annotationId.append(annotationId)
        self.bbx_static_annotationId = np.array(self.bbx_static_annotationId)

        # load metas
        self.build_metas(self.cam2world_dict_00, self.cam2world_dict_01, self.images_list_00, self.images_list_01, self.intersections_dict_00, self.intersections_dict_01, self.box3d_list_00, self.box3d_list_01)

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])   # 3*4行投影矩阵
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)   # 4*4旋转矩阵，右下角为1
            elif line[0] == "S_rect_01:":   # resolution for 1号相机
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def build_metas(self, cam2world_dict_00, cam2world_dict_01, images_list_00, images_list_01, intersection_dict_00, intersection_dict_01, box3d_list_00, box3d_list_01):
        input_tuples = []
        for idx, frameId in enumerate(self.image_ids):
            pose = cam2world_dict_00[frameId]
            pose[:3, 3] = pose[:3, 3] - self.translation    # self.translation是sequence中心，不为0，给第4列平移量减去sequence中心坐标，等价于让sequence中心为0，所有点都围绕sequence中心
            image_path = images_list_00[frameId]
            intersection_path = intersection_dict_00[frameId]
            intersection = np.load(intersection_path)
            intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)   # size(132352, 10, 2)       # 10是最大bbox的数据，2是near far的深度
            intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)  # size(132352, 10, 2)   # dim =1是semanticID，dim=0是globalID
            intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)  # size(132352,10, 4]
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)    # (376, 1408, 3)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])  # [H*W, 6]H*W个像素对应的每根ray的相机原点坐标+ray_d方向坐标
            rays_rgb = image.reshape(-1, 3) # [H*W, 6]
            pseudo_label = cv2.imread(os.path.join(self.pseudo_root, self.scene,self.sequence[-9:-5]+'_{:010}.png'.format(frameId)), cv2.IMREAD_GRAYSCALE)
            pseudo_label = cv2.resize(pseudo_label, (self.W, self.H), interpolation=cv2.INTER_NEAREST)  #[188,704]
            depth = np.loadtxt("datasets/KITTI-360/sgm/{}/depth_{:010}_0.txt".format(self.sequence, frameId))   # [376, 1408]
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            boxes3d =  box3d_list_00[frameId]
            cls_ids = np.array([f('label', item) for item in boxes3d], dtype=np.float32)    # (13,)
            rotys = np.array([f('rot_y', item) for item in boxes3d], dtype=np.float32)      # (13,)
            locations = np.stack([f('locations', item) for item in boxes3d]).astype(np.float32) # (13, 3)
            locations = locations - self.translation
            dimensions = np.stack([f('dimensions', item) for item in boxes3d]).astype(np.float32)   # (13, 3)
            regression = np.stack([f('regression', item) for item in boxes3d]).astype(np.float32)   # (13, 8, 3)
            regression = regression - self.translation

            input_tuples.append((rays, rays_rgb, frameId, intersection, pseudo_label, self.intrinsic_00, 0, depth, cls_ids, rotys, locations, dimensions, regression, pose))
        print('load meta_00 done')
    
        if cfg.use_stereo == True:  # 使用01号相机的图片
            for idx, frameId in enumerate(self.image_ids):
                pose = cam2world_dict_01[frameId]
                pose[:3, 3] = pose[:3, 3] - self.translation
                image_path = images_list_01[frameId]
                intersection_path = intersection_dict_01[frameId]
                intersection = np.load(intersection_path)
                intersection_depths = intersection['arr_0'].reshape(-1, 10, 2).astype(np.float32)
                intersection_annotations = intersection['arr_1'].reshape(-1, 10, 2).astype(np.float32)
                intersection = np.concatenate((intersection_depths, intersection_annotations), axis=2)
                image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])  # 00 和01相机的内参均相同
                rays_rgb = image.reshape(-1, 3)
                pseudo_label = np.zeros_like(pseudo_label)
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                depth = -1 * np.ones_like(image)
                depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                
                boxes3d =  box3d_list_01[frameId]
                cls_ids = np.array([f('label', item) for item in boxes3d], dtype=np.float32)    # (13,)
                rotys = np.array([f('rot_y', item) for item in boxes3d], dtype=np.float32)      # (13,)
                locations = np.stack([f('locations', item) for item in boxes3d]).astype(np.float32) # (13, 3)
                locations = locations - self.translation
                dimensions = np.stack([f('dimensions', item) for item in boxes3d]).astype(np.float32)   # (13, 3)
                regression = np.stack([f('regression', item) for item in boxes3d]).astype(np.float32)   # (13, 8, 3)
                regression = regression - self.translation

                input_tuples.append((rays, rays_rgb, frameId, intersection, pseudo_label, self.intrinsic_01, 1, depth, cls_ids, rotys, locations, dimensions, regression, pose))
            print('load meta_01 done')
        self.metas = input_tuples

    def __getitem__(self, index):
        rays, rays_rgb, frameId, intersection, pseudo_label, intrinsics, stereo_num, depth, cls_ids, rotys, locations, dimensions, regression, pose= self.metas[index]  # 取一张图片的全部光线
        if self.split == 'train':
            rand_ids = np.random.permutation(len(rays)) # 取随机排序的rays的下标
            rays = rays[rand_ids[:cfg.N_rays]]  # 从全部像素对应的光线中取采样 2048rays (2048, 6)
            rays_rgb = rays_rgb[rand_ids[:cfg.N_rays]]  # (2048, 3)
            intersection = intersection[rand_ids[:cfg.N_rays]]  # (2048, 10, 4)
            pseudo_label = pseudo_label.reshape(-1)[rand_ids[:cfg.N_rays]]  # (2048,)
            depth = depth.reshape(-1)[rand_ids[:cfg.N_rays]]    # (2048,)
            
        instance2id, id2instance, semantic2id, instance2semantic = convert_id_instance(intersection)

        ret = {
            'rays': rays.astype(np.float32),
            'rays_rgb': rays_rgb.astype(np.float32),
            'intersection': intersection,
            'intrinsics': intrinsics.astype(np.float32),
            'pseudo_label': pseudo_label,
            'meta': {
                'sequence': '{}'.format(self.sequence),
                'tar_idx': frameId,
                'h': self.H,
                'w': self.W
            },
            'stereo_num': stereo_num,
            'depth': depth.astype(np.float32),
            'instance2id': instance2id,
            'id2instance': id2instance,
            'semantic2id': semantic2id,
            'instance2semantic': instance2semantic,
            'box3d_label' : cls_ids,
            'box3d_rotys' : rotys,
            'box3d_ct_loc' : locations,
            'box3d_hwl' : dimensions,
            'box3d_reg_tgt' : regression,
            'pose' : pose
        }
        return ret

    def __len__(self):
        return len(self.metas)
