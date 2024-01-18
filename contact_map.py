# from collections import defaultdict
# import math
import os
# import sys
# from typing import Iterable
# from cv2 import KeyPoint
import json
import torch
# import util.misc as utils
# from datasets.data_prefetcher import data_prefetcher
from tqdm import tqdm
import numpy as np
# import copy
# from scipy.spatial import procrustes
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as F
# # os.environ["CUB_HOME"] = os.getcwd() + '/cub-1.10.0'
# from pytorch3d.ops.knn import knn_points
# from AIK import AIK_torch as AIK
# import AIK.AIK_config as AIK_config
import pickle
from manopth.manolayer import ManoLayer
import trimesh

from engine import get_NN, get_pseudo_cmap, pixel2cam, rigid_transform_3D_numpy


def main(path):
    obj2idx = {
        "book": 1,
        "espresso": 2,
        "lotion": 3,
        "lotion_spray": 4,
        "milk": 5,
        "cocoa": 6,
        "chips": 7,
        "cappuccino": 8,
    }
    idx2obj = {v: k for k, v in obj2idx.items()}

    GT_obj_vertices_dict = {}
    GT_3D_bbox_dict = {}
    for i in range(1, 9):
        with open(os.path.join("./", "H2O", 'obj_pkl', f'{idx2obj[i]}_2000.pkl'), 'rb') as f:
            vertices = pickle.load(f)
            GT_obj_vertices_dict[i] = vertices
        with open(os.path.join("./", "H2O", 'obj_pkl', f'{idx2obj[i]}_bbox.pkl'), 'rb') as f:
            bbox = pickle.load(f)
            GT_3D_bbox_dict[i] = bbox

    with open(path, "r") as f:
        data = json.load(f)
    images = data['images']
    annos = data['annotations']
    _mano_root = 'mano/models'
    mano_left = ManoLayer(flat_hand_mean=True,
                          side="left",
                          mano_root=_mano_root,
                          use_pca=False,
                          root_rot_mode='axisang',
                          joint_rot_mode='axisang')

    mano_right = ManoLayer(flat_hand_mean=True,
                           side="right",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='axisang',
                           joint_rot_mode='axisang')

    mano_left.cuda()
    mano_right.cuda()

    for img in tqdm(images):
        x, y = img['width'], img['height']
        idx = img['id']
        anno = [an for an in annos if an['image_id'] == idx]
        left_param = [an for an in anno if an['category_id'] == 9]
        right_param = [an for an in anno if an['category_id'] == 10]
        cam_param = np.array(img['cam_param'])
        cam_fx, cam_fy, cam_cx, cam_cy, _, _ = cam_param

        # get object vertices
        object_annot = [an for an in anno if an['category_id'] in obj2idx.values()]
        obj_keyp = object_annot[0]['keypoints']
        obj_label = object_annot[0]['category_id']
        GT_3D_bbox = GT_3D_bbox_dict[obj_label][None]
        obj_keyp = torch.tensor(obj_keyp, dtype=torch.float32)[None].cuda()
        obj_keyp = obj_keyp.reshape(21, 3)
        obj_keyp = obj_keyp * torch.tensor([x, y, 1000]).cuda()
        # obj_keyp = obj_keyp.unsqueeze(0)

        pred_obj_cam = pixel2cam(obj_keyp, (cam_fx, cam_fy), (cam_cx, cam_cy)).unsqueeze(0).detach().cpu().numpy()
        c, R, t = rigid_transform_3D_numpy(GT_3D_bbox * 1000, pred_obj_cam)
        c = torch.from_numpy(c).cuda()
        R = torch.from_numpy(R).cuda()
        t = torch.from_numpy(t).cuda()

        # left hand
        if len(left_param) > 0:
            left_keyp = left_param[0]['keypoints']
            left_keyp = torch.tensor(left_keyp, dtype=torch.float32)[None].cuda()
            left_keyp = left_keyp.reshape(21, 3)
            left_keyp = left_keyp * torch.tensor([x, y, 1000]).cuda()
            left_keyp = left_keyp.unsqueeze(0)
            left_mono_params = left_param[0]['mano_param']
            left_pose, left_shape = torch.as_tensor(left_mono_params[3:51]).unsqueeze(0).cuda(), torch.as_tensor(
                left_mono_params[51:]).unsqueeze(0).cuda()



        # right hand
        if len(right_param) > 0:
            right_keyp = right_param[0]['keypoints']
            right_keyp = torch.tensor(right_keyp, dtype=torch.float32)[None].cuda()
            right_keyp = right_keyp.reshape(21, 3)
            right_keyp = right_keyp * torch.tensor([x, y, 1000]).cuda()
            right_keyp = right_keyp.unsqueeze(0)
            right_mono_params = right_param[0]['mano_param']
            right_pose, right_shape = torch.as_tensor(right_mono_params[3:51]).unsqueeze(0).cuda(), torch.as_tensor(
                right_mono_params[51:]).unsqueeze(0).cuda()

        # concat left and right hand
        if len(left_param) > 0 and len(right_param) > 0:
            hand_kp = torch.cat([left_keyp, right_keyp], dim=0)[None]
        elif len(left_param) == 0:
            hand_kp = right_keyp[None]
        elif len(right_param) == 0:
            hand_kp = left_keyp[None]

        filename = img['file_name']
        dataset = ''
        save_contact_vis_path = os.path.join(f'./contact_vis/{dataset}', filename)
        save_maps_path = os.path.join(f'./contact_maps/{dataset}', filename)


        MANO_LAYER = [mano_left, mano_right] if hand_kp.shape[1] == 2 else [mano_right]

        if len(left_param) > 0 and len(right_param) > 0:
            mano_results = [MANO_LAYER[0](left_pose, left_shape), MANO_LAYER[1](right_pose, right_shape)]
        elif len(left_param) == 0:
            mano_results = [MANO_LAYER[0](right_pose, right_shape)]
        elif len(right_param) == 0:
            mano_results = [MANO_LAYER[0](left_pose, left_shape)]
        # mano_results = [mano_layer(pose_params[:, 48 * i:48 * (i + 1)], opt_tensor_shape) for i, mano_layer in
        #                 enumerate(MANO_LAYER)]
        hand_verts = torch.stack([m[0] for m in mano_results], dim=1)
        j3d_recon = torch.stack([m[1] for m in mano_results], dim=1)

        hand_cam = pixel2cam(hand_kp, (cam_fx.item(), cam_fy.item()), (cam_cx.item(), cam_cy.item()))
        hand_verts = hand_verts - j3d_recon[:, :, :1] + hand_cam[:, :, :1]

        obj_name = idx2obj[obj_label]
        obj_mesh = trimesh.load(f'./H2O/object/{obj_name}/{obj_name}.obj')
        obj_mesh.vertices = (torch.matmul(R[0].detach().cpu(),
                                          torch.tensor(obj_mesh.vertices, dtype=torch.float32).permute(1,
                                                                                                       0) * 1000).permute(1,
                                                                                                                          0) +
                             t[0, None].detach().cpu()).numpy()
        obj_vertices = torch.tensor(obj_mesh.vertices)[None].repeat(1, 1, 1).to(torch.float32).cuda()

        obj_nn_dist_affordance = get_NN(obj_vertices.to(torch.float32), hand_verts.reshape(1, -1, 3).to(torch.float32))
        hand_nn_dist_affordance = torch.stack(
            [get_NN(hand_verts[:, idx].to(torch.float32), obj_vertices.to(torch.float32)) for idx in
             range(hand_verts.shape[1])], dim=1)
        obj_cmap_affordance = get_pseudo_cmap(obj_nn_dist_affordance)
        hand_cmap_affordance = torch.stack(
            [get_pseudo_cmap(hand_nn_dist_affordance[:, idx]) for idx in range(hand_verts.shape[1])], dim=1)

        cmap = plt.cm.get_cmap('plasma')
        obj_v_color = (cmap(obj_cmap_affordance[0].detach().cpu().numpy() * 1000)[:, 0, :-1] * 255)
        hand_v_color = [(cmap(hand_cmap_affordance[0, idx].detach().cpu().numpy() * 1000)[:, 0, :-1] * 255) for
                        idx in range(hand_verts.shape[1])]
        
        obj_mesh = trimesh.Trimesh(vertices=obj_vertices[0].detach().cpu().numpy(), vertex_colors=obj_v_color,
                                   faces=obj_mesh.faces)
        
        hand_mesh = [trimesh.Trimesh(vertices=hand_verts[:, i].detach().cpu().numpy()[0],
                                     faces=(mano_layer.th_faces).detach().cpu().numpy(), vertex_colors=hand_v_color[i])
                     for i, mano_layer in enumerate(MANO_LAYER)]
        
        if not os.path.exists(os.path.dirname(save_contact_vis_path)):
            os.makedirs(os.path.dirname(save_contact_vis_path))

        if not os.path.exists(os.path.dirname(save_maps_path)):
            os.makedirs(os.path.dirname(save_maps_path))

        if len(hand_mesh) == 2:
            trimesh.exchange.export.export_mesh(hand_mesh[0], f'{save_contact_vis_path[:-4]}_left.obj')
            trimesh.exchange.export.export_mesh(hand_mesh[1], f'{save_contact_vis_path[:-4]}_right.obj')
            trimesh.exchange.export.export_mesh(obj_mesh, f'{save_contact_vis_path[:-4]}_obj.obj')
        else:
            trimesh.exchange.export.export_mesh(hand_mesh[0], f'{save_contact_vis_path[:-4]}_right.obj')
            trimesh.exchange.export.export_mesh(obj_mesh, f'{save_contact_vis_path[:-4]}_obj.obj')

        # contactmaps = {"obj": obj_cmap_affordance[0].detach().cpu().numpy(), "hand": hand_cmap_affordance[0].detach().cpu().numpy()}
        # np.savez(f'{save_maps_path[:-4]}_contactmaps.npz', **contactmaps)

        # load contact maps
        # contactmaps = np.load(f'{save_contact_vis_path[:-4]}_contactmaps.npz')
        #
        # print(contactmaps['obj'].shape)
        # print(contactmaps['hand'].shape)
if __name__ == '__main__':
    main("./H2O/H2O_pose_train.json")
    main("./H2O/H2O_pose_test.json")
    main("./H2O/H2O_pose_val.json")