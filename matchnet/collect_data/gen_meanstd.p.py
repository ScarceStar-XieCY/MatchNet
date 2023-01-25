# 均值及标准差
import cv2
import pickle
import os
import numpy as np

def get_mean_std(dirname, norm_info):
    name_list = ["color","gray","depth"]
    for name in name_list:
        if norm_info.get(name,None) is None:
            norm_info[name] = {}
            norm_info[name]["mean"] = []
            norm_info[name]["std"] = []
        final_img = cv2.imread(os.path.join(dirname, "final_"+name+".png"),cv2.IMREAD_UNCHANGED)
        init_img = cv2.imread(os.path.join(dirname, "init_"+name+".png"),cv2.IMREAD_UNCHANGED)
        add_img = (final_img + init_img)/2
        if final_img.ndim == 3 and final_img.shape[2] == 3: # rgb image
            axis = (0,1)
        else:
            axis = None
        mean = np.mean(add_img, axis = axis, dtype= np.float32)
        std = np.std(add_img, axis = axis, dtype= np.float32)
        
        norm_info[name]["mean"].append(mean)
        norm_info[name]["std"].append(std)

    return norm_info

if __name__ == "__main__":
    kit_dir = os.path.join("20230108","datasets","bear")
    dirlist = os.listdir(kit_dir)
    # print(dirlist)
    color_mean_sum = 0
    depth_mean_sum = 0
    color_std_sum = 0
    depth_std_sum = 0
    norm_info = {}
    for dirname in dirlist:
        dirname = os.path.join(os.path.join(kit_dir,dirname))
        if os.path.isdir(dirname):
            # print(dirname)
            # dirname = os.path.join(cfg.data_root,cfg.data_type,dirname)
            norm_info = get_mean_std(dirname,norm_info)

    for key, norm_value in norm_info.items():
        for metric_name, metric_value in norm_value.items():
            metric_array = np.array(metric_value)
            mean_metric = np.mean(metric_array,axis=0)
            if isinstance(mean_metric, np.float32):
                norm_info[key][metric_name] = [mean_metric]
            elif isinstance(mean_metric, np.ndarray):
                mean_metric = mean_metric.squeeze()
                assert mean_metric.ndim == 1
                norm_info[key][metric_name] = list(mean_metric)

    with open(os.path.join(kit_dir,'mean_std.pkl'), 'wb') as file:
        pickle.dump(norm_info, file)

    # with open(os.path.join('./demo', 'mean_std_old.p'), 'rb') as file:
    #     data = pickle.load(file)
    #     print(data)