"""Run the ORB-PE baseline referenced in Section V, A.
"""

import argparse
import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

from matchnet import config
from matchnet.code.utils.pointcloud import transform_xyz
from matchnet.code.utils import common
from tools.matrix import gen_rot_mtx_anticlockwise
from tools.geometry.geometry import estimate_rigid_transform
from sympy.matrices import Matrix, GramSchmidt

logger = logging.getLogger(__name__)

def schmit_matrix(input_matrix,orthonormal=False):
    """orthonormal默认为False，不执行单位化操作,
        orthonormal设为True，执行单位化操作
    """
    all_line = []
    for line in input_matrix:
        all_line.append(Matrix(line))
    # l = [Matrix([3,2,-1]), Matrix([1,3,2]), Matrix([4,1,0])]
    # 注意：将数据转为Matrix格式，否则调用GramSchmidt函数会报错！
    out = GramSchmidt(all_line,orthonormal=orthonormal)
    print(out)
    return np.array(out)


def cal_transfrom(label_info_dict, depth_image):
    delta_angle = label_info_dict["delta_angle"][-1]
    final_point = np.array(label_info_dict["final_point"][-1])
    init_point = np.array(label_info_dict["init_point"][-1])
    
    du, dv = (init_point - final_point).squeeze()
    dz = int(depth_image[init_point[0], init_point[1]]) - int(depth_image[final_point[0], init_point[1]]) #convert to signed from unsigned
    true_transform = np.eye(4)
    true_transform[:3,:3] = gen_rot_mtx_anticlockwise(delta_angle, isdegree=True)
    true_transform[:3,3] = np.array([du,dv,dz])
    # true_transform = schmit_matrix(true_transform)
    return true_transform


def cal_pose(label_info_dict, init, depth_image):
    pose = np.eye(4)
    if init:
        delta_angle = 0
        point_key = "init_point"
    else:
        delta_angle = label_info_dict["delta_angle"][-1]
        point_key = "final_point"
    pose[:3,:3] = gen_rot_mtx_anticlockwise(delta_angle, isdegree=True)
    u,v = np.array(label_info_dict[point_key][-1]).squeeze()
    z = depth_image[u,v]
    pose[:3,3] =np.array([v,u,z])
    # pose = schmit_matrix(pose)
    return pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ORB-PE Baseline on Benchmark")
    parser.add_argument("--debug", type=lambda s: s.lower() in ["1", "true"], default=False)
    args, unparsed = parser.parse_known_args()
    use_color = False
    # instantiate ORB detector
    detector = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    save_dir = os.path.join("../dump/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    kit_poses = {}
    kit_dirs = glob.glob("../datasets" + "/*")
    kit_dirs = list(filter(lambda file_name:file_name[-4:] != ".pkl", kit_dirs))
    for kit_idx, data_dir in enumerate(kit_dirs):
        print(data_dir.split("/")[-1])

        train_dir = os.path.join(data_dir)
        test_dir = os.path.join(data_dir)

        train_foldernames = glob.glob(train_dir + "/*")
        train_foldernames = list(filter(lambda file_name:file_name[-4:] != ".pkl", train_foldernames))
        test_foldernames = glob.glob(test_dir + "/*")
        test_foldernames = list(filter(lambda file_name:file_name[-4:] != ".pkl", test_foldernames))
        
        test_foldernames.sort(key=lambda x: int(x.split("/")[-1]))

        pred_poses = []
        for test_folder in tqdm(test_foldernames, leave=False):
            # # load camera params
            # intr = np.loadtxt(os.path.join(data_dir, "intr.txt"))
            # extr = np.loadtxt(os.path.join(data_dir, "extr.txt"))

            # load test color and depth heightmaps
            color_test = common.colorload(test_folder, init = False,use_color=use_color)
            depth_test = common.depthload(test_folder, init = False)

            # load object mask
            # load info_dict
            info_dict = pickle.load(open(os.path.join(test_folder, "info_dict.pkl"),"rb"))

            obj_idxs_test = info_dict["obj"][-1]
            obj_mask_test = np.zeros_like(color_test,dtype = "uint8")
            obj_mask_test[obj_idxs_test[:, 0], obj_idxs_test[:, 1]] = 1

            # # load initial and final object pose
            # init_pose_test = np.loadtxt(os.path.join(test_folder, "init_pose.txt"))
            # final_pose_test = np.loadtxt(os.path.join(test_folder, "final_pose.txt"))

            # # compute end-effector transform
            init_pose_test = cal_pose(info_dict, init=True, depth_image=depth_test)
            final_pose_test = cal_pose(info_dict, init=False, depth_image=depth_test)
            true_transform = np.linalg.inv(final_pose_test @ np.linalg.inv(init_pose_test))
            # true_transform= cal_transfrom(info_dict,depth_test)
            # true_transform = np.linalg.inv(true_transform)

            # find keypoints and descriptors for current image
            kps_test, des_test = detector.detectAndCompute(color_test, obj_mask_test)

            # loop through train and detect keypoint matches
            matches_train = []
            for i, train_folder in enumerate(train_foldernames):
                # load train data
                color_train = common.colorload(train_folder, init=False, use_color=use_color)
                info_dict_train = pickle.load(open(os.path.join(train_folder, "info_dict.pkl"),"rb"))
                obj_idxs_train = info_dict_train["obj"][-1]
                obj_mask_train = np.zeros_like(color_train, dtype = "uint8")
                obj_mask_train[obj_idxs_train[:, 0], obj_idxs_train[:, 1]] = 1

                # find keypoints in train image
                kps_train, des_train = detector.detectAndCompute(color_train, obj_mask_train)
                if des_train is None:
                    continue

                # brute force match
                matches = bf.match(des_test, des_train)
                if len(matches) < 4:
                    continue
                matches_train.append([i, matches])

            # in case we don't find any matches
            if len(matches_train) == 0:
                pred_poses.append(np.nan)
                continue

            # sort matches by lowest average match distance
            matches_train = sorted(matches_train, key=lambda x: np.mean([y.distance for y in x[1]]))

            # retrieve top match image from database
            idx = matches_train[0][0]
            matches = matches_train[0][1]
            color_train = common.colorload(train_foldernames[idx], init = False,use_color=use_color)
            depth_train = common.depthload(train_foldernames[idx], init = False,)

            info_dict_train = pickle.load(open(os.path.join(train_foldernames[idx], "info_dict.pkl"),"rb"))
            obj_idxs_train = info_dict_train["obj"][-1]
            obj_mask_train = np.zeros_like(color_train, dtype="uint8")
            obj_mask_train[obj_idxs_train[:, 0], obj_idxs_train[:, 1]] = 1
            # init_pose_train = np.loadtxt(os.path.join(train_foldernames[idx], "init_pose.txt"))
            # final_pose_train = np.loadtxt(os.path.join(train_foldernames[idx], "final_pose.txt"))
            init_pose_train = cal_pose(info_dict_train, init=True, depth_image=depth_train)
            final_pose_train = cal_pose(info_dict_train, init=False, depth_image=depth_train)
            transform_train = np.linalg.inv(final_pose_train @ np.linalg.inv(init_pose_train))
            kps_train, des_train = detector.detectAndCompute(color_train, obj_mask_train)

            # plots descriptor matches between query and train
            if args.debug:
                img_debug = cv2.drawMatches(color_test, kps_test, color_train, kps_train, matches, None, flags=2)
                plt.imshow(img_debug)
                plt.show()

            src_pts  = np.float32([kps_test[m.queryIdx].pt for m in matches]).reshape(-1, 2).astype("int")
            dst_pts  = np.float32([kps_train[m.trainIdx].pt for m in matches]).reshape(-1, 2).astype("int")

            # estimate rigid transform from matches projected in 3D
            zs = depth_test[src_pts[:, 1], src_pts[:, 0]].reshape(-1, 1)
            src_xyz = np.hstack([src_pts, zs])
            src_xyz[:, 0] = (src_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
            src_xyz[:, 1] = (src_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
            zs = depth_train[dst_pts[:, 1], dst_pts[:, 0]].reshape(-1, 1)
            dst_xyz = np.hstack([dst_pts, zs])
            dst_xyz[:, 0] = (dst_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
            dst_xyz[:, 1] = (dst_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
            R, t = estimate_rigid_transform(src_xyz, dst_xyz)
            tr = np.eye(4)
            tr[:4, :4] = R
            tr[:3, 3] = t

            # compute estimated transform
            estimated_trans = transform_train @ tr
            pred_poses.append(estimated_trans)

            if args.debug:
                zs = depth_test[obj_idxs_test[:, 0], obj_idxs_test[:, 1]].reshape(-1, 1)
                mask_xyz = np.hstack([obj_idxs_test, zs])
                mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]
                mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
                mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
                mask_xyz = transform_xyz(mask_xyz, estimated_trans)
                mask_xyz[:, 0] = (mask_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
                mask_xyz[:, 1] = (mask_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
                hole_idxs_est = mask_xyz[:, [1, 0]]
                mask_xyz = np.hstack([obj_idxs_test, zs])
                mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]
                mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
                mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
                true_xyz = transform_xyz(mask_xyz, true_transform)
                true_xyz[:, 0] = (true_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
                true_xyz[:, 1] = (true_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
                hole_idxs_true = true_xyz[:, [1, 0]]
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(color_test)
                axes[0].scatter(hole_idxs_true[:, 1], hole_idxs_true[:, 0])
                axes[0].title.set_text('Ground Truth')
                axes[1].imshow(color_test)
                axes[1].scatter(hole_idxs_est[:, 1], hole_idxs_est[:, 0])
                axes[1].title.set_text('Predicted')
                plt.show()

        kit_poses[data_dir.split("/")[-1]] = pred_poses

    with open(os.path.join(save_dir, "ORB-PE_poses.pkl"), "wb") as fp:
        pickle.dump(kit_poses, fp)
    logger.warning("Save pkl at %s",os.path.join(save_dir, "ORB-PE_poses.pkl"))