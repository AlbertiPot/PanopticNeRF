import numpy as np
import math

def cal_hwl(obj3d):
# calculate 3D w,h,l
    center = obj3d.vertices[0]*0.5 + obj3d.vertices[6]*(1-0.5)        # 0-6点构成对角线中点
    front_center = obj3d.vertices[0]*0.5 + obj3d.vertices[3]*(1-0.5)  # 0-3点构成的front面中点
    left_center = obj3d.vertices[0]*0.5 + obj3d.vertices[4]*(1-0.5)   # 0-4点构成的left面中点
    top_center = obj3d.vertices[0]*0.5 + obj3d.vertices[7]*(1-0.5)    # 0-7点构成的top面中点
    
    l = 2* np.linalg.norm(center-front_center)
    w = 2* np.linalg.norm(center-left_center)
    h = 2* np.linalg.norm(center-top_center)
    
    return [h,w,l]

def get_alpha(r_y, z, x):
    '''z,x is the coordinates of the cernter points in camera '''
    alpha = r_y - (math.pi*0.5 - math.atan2(z, x)) # r_y-theta, theta is the angle between camera heading and object heading 
    
    # clip alpha in  [-pi, pi]
    if alpha > math.pi:
        alpha -= 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    
    return alpha

def get_yaw(obj3d, camera, frame):
    '''calculate yaw
        yaw = arctan(dz/dx) where dz and dx is the heading direction
    '''
    # world cood
    center = obj3d.vertices[0]*0.5 + obj3d.vertices[6]*(1-0.5)        # 0-6点构成对角线中点
    front_center = obj3d.vertices[0]*0.5 + obj3d.vertices[3]*(1-0.5)  # 0-3点构成的front面中点
    
    # translate to cam cood
    curr_pose = camera.cam2world[frame]
    T = curr_pose[:3,  3]
    R = curr_pose[:3, :3]
    
    center_cam = camera.world2cam(np.asarray([center]), R, T, inverse=True)
    front_center_cam = camera.world2cam(np.asarray([front_center]), R, T, inverse=True)
    
    xfc, zfc = front_center_cam[0][0], front_center_cam[2][0]
    xc, zc = center_cam[0][0], center_cam[2][0] # 3d box center point cood in camera
    dx, dz = xfc-xc, zfc-zc

    yaw = math.atan2(-dz, dx)
    alpha = get_alpha(yaw, zc, xc)
    
    return yaw, alpha
