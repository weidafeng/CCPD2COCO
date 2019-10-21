# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         make_seg_mask.py
# Author:       wdf
# Date:         2019/10/20
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  为ccpd数据集制作segmentation mask
# Usage： python make_seg_mask.py
#-------------------------------------------------------------------------------


import cv2
from pathlib import Path
import progressbar
from multiprocessing import pool

import os
stuffmap_file_path = os.getcwd() + '/stuffthingmaps/train2017/'  # 在当前目录下新建文件夹，用于存储map中间结果图
if not os.path.exists(stuffmap_file_path):
    os.makedirs(stuffmap_file_path)  # 多层目录

IMAGE_DIR = Path("./map/")  # 要处理的map图片目录
im_files = [f for f in IMAGE_DIR.iterdir()]

# 进度条
w = progressbar.widgets
widgets = ['Progress: ', w.Percentage(), ' ', w.Bar('#'), ' ', w.Timer(),
           ' ', w.ETA(), ' ', w.FileTransferSpeed()]
progress = progressbar.ProgressBar(widgets=widgets)

myPool = pool.Pool(processes=16)  # 并行化处理
for im_file in progress(im_files):
    img_cv = cv2.imread(str(im_file),0)
    cv2.imwrite(stuffmap_file_path + str(im_file.stem)+".png", img_cv)
myPool.close()
myPool.join()
