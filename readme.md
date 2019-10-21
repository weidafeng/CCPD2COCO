# Convert CCPD to COCO

# 目录树：
1. data文件夹-存放ccpd车牌图片
2. pycococreatools文件夹-coco官方utils
3. ccpd_to_coco.py文件-转换代码
4. make_seg_mask.py文件-生成segmentation map文件的代码
5. convert_ccpd_to_coco.sh文件，一步到位

# 运行：
python ccpd_to_coco.py --data data （生成json文件，包括bounding box、mask）
python make_seg_mask.py (生成segmentation map)
或者一步到位：
** bash convert_ccpd_to_coco.sh data **


# 说明：
把ccpd车牌检测数据集转化为coco格式；
转换字段包括：
1. bounding box（左上角、右下角）
2. segmentation box （车牌的四个角点，同时更新了segmentation面积的计算公式）
3. segmentation map （车牌的四个角点形成的平行四边形，比bounding box更精确）
4. label（两类：是否为车牌）

