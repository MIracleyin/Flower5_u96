import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random
import time
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.style.use('seaborn')
mpl.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
mpl.rcParams['axes.unicode_minus']=False     # 正常显示负号

# 训练用的图像尺寸
img_size_net = 128
# 训练的batch大小
batch_size = 32
# 数据库路径
# 获取系统当前路径
path = os.getcwd()
dataset_path = path + '\\train\\dataset\\'
# 各个花的路径
flower_pathes = ['flowers\\daisy', 'flowers\\dandelion', 'flowers\\rose', 'flowers\\sunflower', 'flowers\\tulip']
wordlist = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
# 存放过程和结构的路径
run_path = path + "\\train\\run\\"
if not os.path.exists(run_path):
    os.mkdir(run_path)
# 存放转换后的tf数据集的路径
dataset_tf_path = run_path + 'flowersTf.tfrecords'


# to tfrecords
tick_begin = time.time()
img_cnt = int(0)
label_cnt = int(0)
with tf.io.TFRecordWriter(dataset_tf_path) as writer:
    for sort_path in flower_pathes:    
        flower_list = os.listdir(dataset_path + sort_path)
        for img_name in flower_list:
            img_path = dataset_path + sort_path + "/" + img_name
            img = cv2.imread(img_path)  
            img_scale = cv2.resize(img,(img_size_net, img_size_net), interpolation = cv2.INTER_CUBIC)
            if not img is None:
                feature = {
                    'img1':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_scale.tostring()])),
                    'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label_cnt]))
#                     'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label_cnt]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                # 每隔50张打印一张图片
                if img_cnt % 1000 == 0:
                    print('The ', str(img_cnt), ' image')
                    plt.imshow(cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB))
                    plt.show()
                img_cnt += 1
        label_cnt = label_cnt + 1
    writer.close()   
tick_end = time.time()
print('Generate the dataset complete! Experied ', str(tick_end - tick_begin))
print('The dataset is ', dataset_tf_path)