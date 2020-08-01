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
tf.enable_eager_execution()

# 训练用的图像尺寸
img_size_net = 128
# 训练的batch大小
batch_size = 32
path = os.getcwd()
run_path = path + "\\train\\run\\"
wordlist = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']

def read_and_decode(example_proto):
    '''
    从TFrecord格式文件中读取数据

    '''
    image_feature_description  = {
        'img1':tf.io.FixedLenFeature([],tf.string),
        'label':tf.io.FixedLenFeature([1], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_proto, image_feature_description)
    img1 = tf.io.decode_raw(feature_dict['img1'], tf.uint8)
    label = feature_dict['label']
    return img1, label
path = os.getcwd()

dataset_tf_path = path + "\\Train\\run\\flowersTf.tfrecords"
dataset_nums = 4300

# 1. 读取数据集
dataset = tf.data.TFRecordDataset(dataset_tf_path)
dataset = dataset.map(read_and_decode)

# 2. 随机打印8个测试图像
dataset = dataset.shuffle(buffer_size=dataset_nums)
dataSet = np.array([x1 for x1 in dataset.take(10)])
dataSet_img = np.array([x1[0].numpy() for x1 in dataSet])
dataSet_img = dataSet_img.reshape((-1,img_size_net,img_size_net, 3)) / ((np.float32)(255.))
dataSet_label = np.array([x1[1].numpy()[0] for x1 in dataSet]) 
fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15,15)
l = 0
for i in range(5):
    for j in range(2):
        ax[i, j].imshow(cv2.cvtColor(dataSet_img[l], cv2.COLOR_BGR2RGB))
        ax[i, j].set_title(wordlist[dataSet_label[l]])
        l += 1
plt.tight_layout()
#plt.show()

# 1. 打乱数据集
dataset = dataset.shuffle(buffer_size=dataset_nums)
# 2. 抓取数据集
dataSet = np.array([x1 for x1 in dataset.take(dataset_nums)])
dataSet_img = np.array([x1[0].numpy() for x1 in dataSet])
dataSet_img = dataSet_img.reshape((-1,img_size_net,img_size_net, 3)) / ((np.float32)(255.))
dataSet_label = np.array([x1[1].numpy()[0] for x1 in dataSet]) 
# 3. 分离训练集和测试集
trainSet_num = int(0.75 * dataset_nums)
trainSet_img = dataSet_img[0 : trainSet_num, :, :, :]
testSet_img = dataSet_img[trainSet_num : , :, :, :]
trainSet_label = dataSet_label[0 : trainSet_num]
testSet_label = dataSet_label[trainSet_num : ]

# 3. 统计各种训练集中各种样本的数量
print('数据集中各个样本的数量：')
l = []
for x in dataSet_label:
    l.append(wordlist[x])
plt.hist(l, rwidth=0.5)
plt.show()

input_tensor = tf.keras.layers.Input(shape=(img_size_net,img_size_net,3), name="x_input")
base_model = tf.keras.applications.InceptionV3(weights='imagenet', 
                                               include_top=False, 
                                               input_tensor=input_tensor, 
                                               input_shape=(img_size_net,img_size_net,3))

#base_model.summary()    #打印网络结构

base_model = tf.keras.Model(inputs=input_tensor, outputs=base_model.get_layer('activation_74').output)

# 添加全局平均池化层
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# 添加一个分类器，假设我们有200个类
predictions = tf.keras.layers.Dense(5, activation='softmax', name='y_out')(x)

# 构建我们需要训练的完整模型
model = tf.keras.Model(inputs=input_tensor, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in model.layers[:195]:
    layer.trainable = False
for layer in model.layers[195:]:
    layer.trainable = True

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()    #打印网络结构
history = model.fit(trainSet_img, trainSet_label, 
                    batch_size=batch_size, 
                    epochs=20, 
                    validation_data=(testSet_img, testSet_label)
                    )
# 我们锁住前面135层，然后放开之后的层。
for layer in model.layers[:135]:
    layer.trainable = False
for layer in model.layers[135:]:
    layer.trainable = True
# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调    
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(trainSet_img, trainSet_label, 
                    batch_size=batch_size, 
                    epochs=20, 
                    validation_data=(testSet_img, testSet_label)
                    )
model.save_weights(run_path + "model_weight.h5")
json_config = model.to_json()
with open(run_path + 'model_config.json', 'w') as json_file:
    json_file.write(json_config)