# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         ccpd_to_coco.py
# Author:       wdf
# Date:         2019/10/18
# IDE：         PyCharm
# Description:  把ccod车牌检测数据集转化为coco格式（bbox、mask、stuffthingmaps）
# Usage：
#     python ccpd_to_coco.py --data "./data"
#-------------------------------------------------------------------------------

import datetime
import json
import cv2
from random import randint
import numpy as np
from pathlib import Path
from PIL import Image

from multiprocessing import pool
from pycococreatortools import pycococreatortools

import os
file_path = os.getcwd() + '/map/'  # 在当前目录下新建map文件夹，用于存储map中间结果图
if not os.path.exists(file_path):
    os.mkdir(file_path)

import argparse
parser = argparse.ArgumentParser()
## 从命令行指定kjdz文件夹
'''目录树如下
|- KJDZ
|--  IMAGES
|----    *.jpg
|--  ANNOTATIONS
|----    *.txt
'''
parser.add_argument("--data",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain all the images")

args = parser.parse_args()

IMAGE_DIR = Path(args.data)


INFO = {
    "description": "CCPD Dataset in COCO Format",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Tristan, Onlyyou intern",
    "date_created": datetime.datetime.utcnow().isoformat(' ')  # 显示此刻时间，格式：'2019-04-30 02:17:49.040415'
}

LICENSES = [
    {
        "id": 1,
        "name": "ALL RIGHTS RESERVED",
        "url": ""
    }
]
# 初始化类别（背景）
CATEGORIES = [
    {
        'id': 1,
        'name': 'license plate',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'background',
        'supercategory': 'shape',
    }
]


# 根据车牌的四个角点绘制精确的segmentation map
def make_seg_mask(map, segmentions,color=(0,255,0)):
    c = np.array([[segmentions]], dtype=np.int32)
    cv2.fillPoly(map, c, color)

def random_color(class_id):
    '''预定义12种颜色，基本涵盖kjdz所有label类型
    颜色对照网址：https://tool.oschina.net/commons?type=3'''
    colorArr = [(255,0,0), # 红色
                (255,255,0), # 黄色
                (0, 255, 0), # 绿色
                (0,0,255), # 蓝色
                (160, 32, 240), # 紫色
                (165, 42, 42), # 棕色
                (238, 201, 0), # gold
                (255, 110, 180), # HotPink1
                (139, 0 ,0), #DarkRed
                (0 ,139 ,139),#DarkCyan
                (139, 0 ,139),#	DarkMagenta
                (0 ,0 ,139) # dark blue
                ]
    if class_id < 11:
        return colorArr[class_id]
    else: # 如有特殊情况，类别数超过12，则随机返回一个颜色
        rm_col = (randint(0,255),randint(0,255),randint(0,255))
        return rm_col

# 获取 bounding-box， segmentation 信息
# 输入：image path
# 返回：
#   bounding box
#   four locations

def get_info(im_file):
    img_name = str(im_file)
    lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3] # label: '16_2_32_30_25_29_6'
    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [leftUp, rightDown] = [[float(eel) for eel in el.split('&')] for el in iname[2].split('_')] # bounding box
    height = rightDown[1]-leftUp[1]
    width = rightDown[0]-leftUp[0]
    left = leftUp[0]
    top = leftUp[1]
    segmentation = [[float(eel) for eel in el.split('&')] for el in iname[3].split('_')] # four vertices locations

    return [left, top, width, height], segmentation


# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3):
        return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    #for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num):
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)


def main():
    # coco lable文件（如training2017.json）需要存储的信息
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 初始化id（以后依次加一）
    image_id = 1
    annotation_id = 1

    # 加载图片信息
    im_files = [f for f in IMAGE_DIR.iterdir()]
    im_files.sort(key=lambda f: f.stem,reverse=True)  # 排序，防止顺序错乱、数据和标签不对应
    # print("im-length:",len(im_files),"\n im_files：",im_files)


    for im_file in im_files:
        # 写入图片信息（id、图片名、图片大小）,其中id从1开始
        image = Image.open(im_file)
        im_info = pycococreatortools.create_image_info( image_id, im_file.name, image.size) # 图片信息
        coco_output['images'].append(im_info) # 存储图片信息（id、图片名、大小）


        myPool = pool.Pool(processes=16)  # 并行化处理
        annotation_info_list = []  # 存储标注信息

        # 用于制作stuff-thing map
        img_cv = cv2.imread(str(im_file)) #调用opencv读取，方便后面使用opencv绘制mask、保存结果
        rectangle = np.zeros(img_cv.shape[0:3], dtype="uint8") # 新建空白图像
        # 使用白色填充图片区域,默认为0-黑色,255-白色
        rectangle.fill(125) # 125 灰色

        # 处理label信息， 包括左上角、右下角、四个角点（用于分割）
        bounding_box, segmentation = get_info(im_file)
        class_id = 1  # id 为数字形式，如 1,此时是list形式，后续需要转换 # 指定为1，因为只有”是车牌“这一类

        # 显示日志
        print(bounding_box, segmentation)

        # 制作stuff-thing map
        color = random_color(class_id)  # 得到当前类别的颜色，保证每幅图像里每一类的颜色都相同
        make_seg_mask(rectangle,segmentation, color)

        # area = bounding_box[-1] * bounding_box[-2]  # 当前bounding-box的面积,宽×高
        area = compute_polygon_area(segmentation) # 当前segmentation的面积（比bounding box更精确）

        an_infos = pycococreatortools.mask_create_annotation_info(annotation_id=annotation_id, image_id=image_id,
                                                                  category_id=class_id, area=area,
                                                                  image_size=image.size, bounding_box=bounding_box,
                                                                  segmentation=segmentation)
        annotation_info_list.append(an_infos)
        cv2.imwrite(file_path+str(im_file.stem)+".png",rectangle)

        myPool.close()
        myPool.join()

        # 上面得到单张图片的所有bounding-box信息，接下来每单张图片存储一次
        for annotation_info in annotation_info_list:
            if annotation_info is not None:
                coco_output['annotations'].append(annotation_info)
        image_id += 1

    # 保存成json格式
    print("[INFO] Storing annotations json file...")
    output_json = Path(f'ccpd_annotations.json')
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(coco_output, f)
    print("[INFO] Annotations JSON file saved in：", str(output_json))


if __name__ == "__main__":
    main()
