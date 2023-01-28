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
    color_mean_sum = 0
    depth_mean_sum = 0
    color_std_sum = 0
    depth_std_sum = 0
    norm_info = {}

    root_dir = os.path.join("20230108","datasets")
    for root_name, dir_list, file_list in os.walk(root_dir):
        if "bear" in dir_list:
            continue
        for time_dir in dir_list:
            dirname = os.path.join(root_name, time_dir)
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

    with open(os.path.join(root_dir,'mean_std.pkl'), 'wb') as file:
        pickle.dump(norm_info, file)

    # with open(os.path.join(root_dir,'mean_std.pkl'), 'rb') as file:
    #     a = pickle.load(file)
