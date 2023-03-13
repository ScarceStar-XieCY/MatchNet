"""Run the Form2Fit models on the benchmark.
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm
from scipy.stats import mode
from tools.image_mask.mask_process import mask2coord
from skimage.draw import circle_perimeter
from matchnet.code.ml.models import CorrespondenceNet
from matchnet.code.ml.dataloader import get_corr_loader
from matchnet.code.utils import misc, ml
from matchnet.code.utils.pointcloud import transform_xyz
from tools.matrix import gen_rot_mtx_anticlockwise
import logging

logger = logging.getLogger(__file__)
INTERVAL = 10
RADIUS = 5
PRED_ROTATIONS = "pred_rotations"
GT_ROT = "gt_rot"
KIT_IDXS_ALL = "kit_idxs_all"
OBJ_IDXS_ALL = "obj_idxs_all"
COORD_NAMES = ["obj_uniform","obj_ccircle"]

def get_circle_point(shape, uv_center, radius):
    """Get all point of one circle."""
    uv_center = uv_center[0] # first of one batch
    mask = np.zeros(shape,dtype = "uint8")
    assert uv_center[0] < shape[0] and uv_center[0] >= 0
    assert uv_center[1] < shape[1] and uv_center[1] >= 0
    rr, cc = circle_perimeter(uv_center[0], uv_center[1], radius)
    mask[rr,cc] = 255
    coord = np.stack([rr,cc],axis = 1)
    cv2.fillConvexPoly(mask,coord[:,::-1],255)
    circle_coord = mask2coord(mask, need_xy=False)
    return circle_coord


def put_img_array_to_dict(norm_info, imgs,data_info_dict):
    color_mean,color_std,depth_mean,depth_std = norm_info

    color_channel =  len(color_mean)
    
    color_kit = ml.tensor2ndarray(imgs[:, :color_channel, :], [color_mean, color_std], True).squeeze()
    depth_kit = ml.tensor2ndarray(imgs[:, color_channel, :], [depth_mean, depth_std], False).squeeze()
    color_obj = ml.tensor2ndarray(imgs[:, color_channel+1:2*color_channel+1, :], [color_mean, color_std], True).squeeze()
    depth_obj = ml.tensor2ndarray(imgs[:, 2*color_channel+1, :], [depth_mean, depth_std], False).squeeze()

    data_info_dict["color_kit"] = color_kit
    data_info_dict["depth_kit"] = depth_kit
    data_info_dict["color_obj"] = color_obj
    data_info_dict["depth_obj"] = depth_obj



def valid_one_data(dloader_output, model, device, num_subsample,debug, norm_info):
    imgs, labels, kit_center, obj_center = dloader_output
    # remove padding from labels
    imgs, labels = imgs.to(device), labels.to(device)

    data_info_dict = {}
    label = labels[0]
    mask = torch.all(label == torch.LongTensor([999]).repeat(6).to(device), dim=1)
    label = label[~mask]

    # extract correspondences from label
    source_idxs = label[:, 0:2]
    target_idxs = label[:, 2:4]
    rot_idx = label[:, 4]
    is_match = label[:, 5]
    correct_rot = rot_idx[0]
    mask = (is_match == 1) & (rot_idx == correct_rot)
    kit_idxs = source_idxs[mask]
    obj_idxs = target_idxs[mask]

    if debug:
        put_img_array_to_dict(norm_info,imgs,data_info_dict)
        kit_idxs_all = kit_idxs.clone()
        obj_idxs_all = obj_idxs.clone()
        data_info_dict["kit_idxs_all"] =kit_idxs_all
        data_info_dict["obj_idxs_all"] =obj_idxs_all

    if num_subsample is not None:
        kit_idxs = kit_idxs[::int(num_subsample)]
        obj_idxs = obj_idxs[::int(num_subsample)]


    
    H, W = imgs.shape[2:]
    obj_center_coord = get_circle_point((H,W),obj_center,RADIUS)
    obj_center_coord = torch.LongTensor(obj_center_coord).to(device)


    # compute kit and object descriptor maps
    with torch.no_grad():
        outs_s, outs_t = model(imgs, *kit_center[0])
    out_s = outs_s[0]
    D, H, W = outs_t.shape[1:]
    out_t = outs_t[0:1]
    out_t_flat = out_t.view(1, D, H * W).permute(0, 2, 1)

    
    # loop through ground truth correspondences
    # obj_uvs = []
    # predicted_kit_uvs = []
    data_info_dict[GT_ROT] = correct_rot.item()
    corrd_idxs = [obj_idxs, obj_center_coord]
    for coord_name, coord_idxs in zip(COORD_NAMES,corrd_idxs):
        pred_rotations = []
        for corr_idx, (u, v) in enumerate(coord_idxs): # all point on one obj
            idx_flat = u*W + v
            target_descriptor = torch.index_select(out_t_flat, 1, idx_flat).squeeze(0)
            outs_s_flat = out_s.view(20, out_s.shape[1], H*W)
            target_descriptor = target_descriptor.unsqueeze(0).repeat(20, H*W, 1).permute(0, 2, 1)
            diff = outs_s_flat - target_descriptor
            l2_dist = diff.pow(2).sum(1).sqrt()
            # heatmaps = l2_dist.view(l2_dist.shape[0], H, W).cpu().numpy()
            predicted_best_idx = l2_dist.min(dim=1)[0].argmin()
            pred_rotations.append(predicted_best_idx.item())
            # min_val = heatmaps[predicted_best_idx].argmin()
            # u_min, v_min = np.unravel_index(min_val, (H, W))
            # predicted_kit_uvs.append([u_min.item(), v_min.item()])
            # obj_uvs.append([u.item(), v.item()])
        
        data_info_dict[PRED_ROTATIONS + "_" + coord_name] = pred_rotations
        data_info_dict["point_num" + "_" + coord_name] = len(coord_idxs)


    return data_info_dict



def validation_correspondence(dloader,model,device, num_subsample=None,interval=1,debug=False):
    
    color_mean = dloader.dataset.c_mean
    color_std = dloader.dataset.c_std
    depth_mean = dloader.dataset.d_mean
    depth_std = dloader.dataset.d_std
    norm_info = [color_mean,color_std,depth_mean,depth_std]
    model.eval()
    prec_dict = {"acc":{},"ap":{}}
    for idx, dloader_output in enumerate(tqdm(dloader)):
        if idx % interval != 0:
            continue
        data_info_dict = valid_one_data(dloader_output,model,device, num_subsample,debug, norm_info)
        correct_rot = data_info_dict[GT_ROT]
        for coord_name in COORD_NAMES:
            if prec_dict["ap"].get(coord_name, None) is None:
                prec_dict["ap"][coord_name] = []
                prec_dict["acc"][coord_name] = 0
            
            pred_rotations = data_info_dict[PRED_ROTATIONS + "_" + coord_name]
            point_num = data_info_dict["point_num" + "_" + coord_name]
            pred_rotations_array = np.array(pred_rotations)
        
            # each obj hav one prec
            tp_seq = np.array(pred_rotations_array == correct_rot)
            tp = np.sum(tp_seq)
            prec = tp / point_num
            
            # compute rotation majority
            best_rot = mode(pred_rotations,axis=None,keepdims=False)[0]
            mode_num = np.sum(pred_rotations_array == best_rot)
            mode_ratio = mode_num / point_num
            # prec_list.append([idx, correct_rot, prec, mode_ratio])
            if debug:
                prec_info = [idx,correct_rot == best_rot, correct_rot, best_rot, prec,mode_ratio]
            else:
                prec_info = prec
            prec_dict["ap"][coord_name].append(prec_info)

            if best_rot == correct_rot:
                prec_dict["acc"][coord_name] += 1

    for coord_name in COORD_NAMES:        
        prec_dict["acc"][coord_name] = prec_dict["acc"][coord_name] * interval/ len(dloader)
        prec_dict["ap"][coord_name] = np.mean(prec_dict["ap"][coord_name]) if not debug else prec_dict["ap"][coord_name]
        # ap = np.mean(prec_list)
    return prec_dict


def recursive_read(input, all_key):
    if isinstance(input, dict):
        for key, value in input.items():
            recursive_read(value, all_key + "_" + key)
    else:
        print(all_key, input)

def main(args):
    print("evaluate ", args.model_path, "at dataset", args.foldername)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    save_dir = os.path.join("../dump/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_channels = 4
    dloader = get_corr_loader(
        args.foldername,
        batch_size=1,
        sample_ratio=1,
        shuffle=False,
        dtype=args.dtype,
        num_rotations=20,
        num_workers=2,
        stateless=True,
        augment=False,
        use_color = True,
        num_channels=num_channels,
        background_subtract=None,
    )


    color_mean = dloader.dataset.c_mean
    color_std = dloader.dataset.c_std
    depth_mean = dloader.dataset.d_mean
    depth_std = dloader.dataset.d_std

    # load model
    model = CorrespondenceNet(num_channels, 64, 20).to(device)
    # state_dict = torch.load(os.path.join(config.weights_dir, "matching", args.foldername + ".tar"), map_location=device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    estimated_poses = []
    correct = 0

    pred_dict = validation_correspondence(dloader,model,device,args.subsample,args.interval)
    recursive_read(pred_dict,"")
    # for idx, (imgs, labels, center) in enumerate(tqdm(dloader)):
    #     # print("{}/{}".format(idx+1, len(dloader)))
    #     if idx % 10 != 0:
    #         continue
    #     imgs, labels = imgs.to(device), labels.to(device)

    #     # remove padding from labels
    #     label = labels[0]
    #     mask = torch.all(label == torch.LongTensor([999]).repeat(6).to(device), dim=1)
    #     label = label[~mask]

    #     # extract correspondences from label
    #     source_idxs = label[:, 0:2]
    #     target_idxs = label[:, 2:4]
    #     rot_idx = label[:, 4]
    #     is_match = label[:, 5]
    #     correct_rot = rot_idx[0]
    #     mask = (is_match == 1) & (rot_idx == correct_rot)
    #     kit_idxs = source_idxs[mask]
    #     obj_idxs = target_idxs[mask]
    #     kit_idxs_all = kit_idxs.clone()
    #     obj_idxs_all = obj_idxs.clone()
    #     if args.subsample is not None:
    #         kit_idxs = kit_idxs[::int(args.subsample)]
    #         obj_idxs = obj_idxs[::int(args.subsample)]

    #     H, W = imgs.shape[2:]

    #     # compute kit and object descriptor maps
    #     with torch.no_grad():
    #         outs_s, outs_t = model(imgs, *center[0])
    #     out_s = outs_s[0]
    #     D, H, W = outs_t.shape[1:]
    #     out_t = outs_t[0:1]
    #     out_t_flat = out_t.view(1, D, H * W).permute(0, 2, 1)

    #     color_kit = ml.tensor2ndarray(imgs[:, 0, :], [color_mean, color_std], True).squeeze()
    #     color_obj = ml.tensor2ndarray(imgs[:, 2, :], [color_mean, color_std], True).squeeze()
    #     depth_kit = ml.tensor2ndarray(imgs[:, 1, :], [depth_mean, depth_std], False).squeeze()
    #     depth_obj = ml.tensor2ndarray(imgs[:, 3, :], [depth_mean, depth_std], False).squeeze()
    #     if args.debug:
    #         kit_idxs_np = kit_idxs_all.detach().cpu().numpy().squeeze()
    #         obj_idxs_np = obj_idxs_all.detach().cpu().numpy().squeeze()
    #         fig, axes = plt.subplots(1, 2)
    #         color_kit_r = misc.rotate_img(color_kit, -(360/20)*correct_rot, center=(center[0][1], center[0][0]))
    #         axes[0].imshow(color_kit_r)
    #         axes[0].scatter(kit_idxs_np[:, 1], kit_idxs_np[:, 0], c='r')
    #         axes[1].imshow(color_obj)
    #         axes[1].scatter(obj_idxs_np[:, 1], obj_idxs_np[:, 0], c='b')
    #         for ax in axes:
    #             ax.axis('off')
    #         plt.show()

    #     # loop through ground truth correspondences
    #     obj_uvs = []
    #     predicted_kit_uvs = []
    #     rotations = []
        
    #     for corr_idx, (u, v) in enumerate(obj_idxs):
    #         idx_flat = u*W + v
    #         target_descriptor = torch.index_select(out_t_flat, 1, idx_flat).squeeze(0)
    #         outs_s_flat = out_s.view(20, out_s.shape[1], H*W)
    #         target_descriptor = target_descriptor.unsqueeze(0).repeat(20, H*W, 1).permute(0, 2, 1)
    #         diff = outs_s_flat - target_descriptor
    #         l2_dist = diff.pow(2).sum(1).sqrt()
    #         heatmaps = l2_dist.view(l2_dist.shape[0], H, W).cpu().numpy()
    #         predicted_best_idx = l2_dist.min(dim=1)[0].argmin()
    #         rotations.append(predicted_best_idx.item())
    #         # if predicted_best_idx == correct_rot:
    #         #     correct += 1
    #         min_val = heatmaps[predicted_best_idx].argmin()
    #         u_min, v_min = np.unravel_index(min_val, (H, W))
    #         predicted_kit_uvs.append([u_min.item(), v_min.item()])
    #         obj_uvs.append([u.item(), v.item()])
        

    #     # compute rotation majority
    #     best_rot = mode(rotations,axis=None,keepdims=False)[0]
    #     if best_rot == correct_rot.item():
    #         correct += 1

    #     # eliminate correspondences with rotation different than mode
    #     select_idxs = np.array(rotations) == best_rot
    #     predicted_kit_uvs = np.array(predicted_kit_uvs)[select_idxs]
    #     obj_uvs = np.array(obj_uvs)[select_idxs]

    #     # use predicted correspondences to estimate affine transformation
    #     src_pts = np.array(obj_uvs)[:, [1, 0]]
    #     dst_pts = np.array(predicted_kit_uvs)
    #     dst_pts = misc.rotate_uv(dst_pts, (360/20)*best_rot, H, W, cxcy=center[0])[:, [1, 0]]

    #     # compose transform
    #     zs = depth_obj[src_pts[:, 1], src_pts[:, 0]].reshape(-1, 1)
    #     src_xyz = np.hstack([src_pts, zs])
    #     # src_xyz[:, 0] = (src_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
    #     # src_xyz[:, 1] = (src_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
    #     zs = depth_kit[dst_pts[:, 1], dst_pts[:, 0]].reshape(-1, 1)
    #     dst_pts[:, 0] += W
    #     dst_xyz = np.hstack([dst_pts, zs])
    #     # dst_xyz[:, 0] = (dst_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
    #     # dst_xyz[:, 1] = (dst_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
    #     m1 = np.eye(4)
    #     dst_xyz[:, 2] = src_xyz[:, 2]
    #     m1[:3, 3] = np.mean(dst_xyz, axis=0) - np.mean(src_xyz, axis=0)
    #     m2 = np.eye(4)
    #     m2[:3, 3] = -np.mean(dst_xyz, axis=0)
    #     m3 = np.eye(4)
    #     m3[:3, :3] = gen_rot_mtx_anticlockwise(np.radians(-(360/20)*best_rot))
    #     m4 = np.eye(4)
    #     m4[:3, 3] = np.mean(dst_xyz, axis=0)
    #     estimated_pose = m4 @ m3 @ m2 @ m1
    #     estimated_poses.append(estimated_pose)

    #     # plot
    #     if args.debug:
    #         img = np.zeros((H, W*2))
    #         img[:, :W] = color_obj
    #         img[:, W:] = color_kit
    #         zs = depth_obj[obj_idxs_np[:, 0], obj_idxs_np[:, 1]].reshape(-1, 1)
    #         mask_xyz = np.hstack([obj_idxs_np, zs])
    #         mask_xyz[:, [0, 1]] = mask_xyz[:, [1, 0]]
    #         # mask_xyz[:, 0] = (mask_xyz[:, 0] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[0, 0]
    #         # mask_xyz[:, 1] = (mask_xyz[:, 1] * config.HEIGHTMAP_RES) + config.VIEW_BOUNDS[1, 0]
    #         mask_xyz = transform_xyz(mask_xyz, estimated_pose)
    #         # mask_xyz[:, 0] = (mask_xyz[:, 0] - config.VIEW_BOUNDS[0, 0]) / config.HEIGHTMAP_RES
    #         # mask_xyz[:, 1] = (mask_xyz[:, 1] - config.VIEW_BOUNDS[1, 0]) / config.HEIGHTMAP_RES
    #         hole_idxs = mask_xyz[:, [1, 0]]
    #         plt.imshow(img)
    #         plt.scatter(hole_idxs[:, 1], hole_idxs[:, 0])
    #         plt.show()
    # print("acc: {}".format(correct / len(dloader)))
    # with open(os.path.join(save_dir, "{}_poses.pkl".format(args.foldername)), "wb") as fp:
    #     pickle.dump(estimated_poses, fp)
    # logger.warning("Save pkl at %s",os.path.join(save_dir, "{}_poses.pkl".format(args.foldername)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Form2Fit Matching Module on Benchmark")
    parser.add_argument("--foldername", type=str, help="The name of the dataset.",default="../datasets_mix0128")
    parser.add_argument("--model_path", type=str, help="Path to model.",default="matchnet/code/ml/savedmodel/mix0128_2/corrs_epoch199.pth")
    parser.add_argument("--dtype", type=str, default="train")
    parser.add_argument("--subsample", type=int, default=16)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--debug", type=lambda s: s.lower() in ["1", "true"], default=False)
    args, unparsed = parser.parse_known_args()
    main(args)