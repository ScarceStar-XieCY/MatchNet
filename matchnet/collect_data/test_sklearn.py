from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans,MiniBatchKMeans,MeanShift,DBSCAN,OPTICS
from matplotlib import pyplot
import cv2
import os
import numpy as np

# 定义数据集
data_root = '20221029test'
data_type = "train"
color_image_in = cv2.imread(os.path.join(data_root,data_type, f"depth1.png"),cv2.IMREAD_GRAYSCALE)
color_image_out = cv2.imread(os.path.join(data_root,data_type, f"color2.png"))
h,w = color_image_in.shape[:2]
color_image_in = cv2.resize(color_image_in,(w//4, h//4))
h,w = color_image_in.shape[:2]
idx_array = np.mgrid[0:h:1,0:w:1]
idx_array = np.reshape(idx_array,(-1,h*w))
color_point = np.reshape(color_image_in,(h*w))
color_point = np.stack([color_point,idx_array[0],idx_array[1]], axis =1)
# color_point, _ = make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

# 定义模型
model = KMeans(n_clusters=4)
# model = MiniBatchKMeans(n_clusters=7)
# model = DBSCAN(eps=0.1, min_samples=150)
# model = MeanShift() # too slow
# model = OPTICS(eps=0.8, min_samples=50)
# 模型拟合
# model.fit(color_point)
# # 为每个示例分配一个集群

# yhat = model.predict(color_point)
yhat = model.fit_predict(color_point)
yhat = np.reshape(yhat,(h,w))
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
ax = pyplot.subplot(1,1,1)
for cluster in clusters:
# 获取此群集的示例的行索引
    img_idx = where(yhat == cluster)
    # 创建这些样本的散布
    ax.scatter(img_idx[1], h - img_idx[0])
# 绘制散点图
pyplot.show()

