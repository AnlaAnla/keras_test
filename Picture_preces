import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
from PIL import Image
import time

flow_path = '../../.keras/datasets/flower_photos'
data_root = pathlib.Path(flow_path)

# 迭代输出该文件下的子文件目录名
# print(flow_path)
# for item in data_root.iterdir():
    # print(item)

# 读取该目录的子目录下所有图片
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)

# -------------------------------------------
# 图片转换为数组后作为文件保存,自己的方法，下面有用tf的方法
# data = np.empty([4,180,180,3])
#
# for i in range(1,4):
#     img = Image.open(all_image_paths[i])
#     img = Image.Image.resize(img,(180,180))
#     img = np.array(img)/255.0
#     data[i] = img
# np.save('D:/ML/model/one_np/data_1',data)
# data = np.load('D:/ML/model/one_np/data_1.npy')
# print(np.shape(data))
#
# plt.figure(10)
# for i in range(1,4):
#     plt.subplot(2,2,i)
#     plt.imshow(data[3])
# plt.show()

# -------------------------------------
# 最后将其以 ‘ CC-BY’分割为列表
# attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
# attributions = [line.split(' CC-BY') for line in attributions]
# attributions = dict(attributions)
# def caption_image(image_path):
    # image_rel = pathlib.Path(image_path).relative_to(data_root)
    # return "Image (CC BY 2.0) " + '-'.join(attributions[str(image_rel)].split('-')[:-1])
# print(caption_image(all_image_paths[0]))
# 读取第77行，0是后半部分路径，下一行用的pathlib路径合并方法
# aa = Image.open(aa)
# plt.imshow(aa)
# plt.show()
# print(type(aa))
# --------------------------------------
# 下面这顿操作太骚气了！！！！文件标签处理一气呵成！
label_name = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# print(label_name)
label_to_index = dict((name, index) for index,name in enumerate(label_name))
# print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

# print(all_image_labels[:10])

#----------------------------------------------------
# 用tf读取处理图片。。居然包含PIL的Image的功能。。。
# img_path = all_image_paths[0]
# img_raw = tf.io.read_file(img_path)
# print(repr(img_raw)[:100]+"...")
# img_tensor = tf.image.decode_image(img_raw)
# print(img_tensor.shape)
# print(img_tensor.dtype)
# -----------------------------------------------------



def preprocess(img_final):
    # ?????
    img_final = tf.image.decode_jpeg(img_final, channels=3)
    img_final = tf.image.resize(img_final,[180,180])/255.0
    return img_final

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess(image)

# 展示某张图片---------------------------
# image_parh = all_image_paths[3]
# label = all_image_labels[3]
# plt.imshow(load_and_process_image(image_parh))
# plt.grid(False)
# plt.title(label_name[label].title())
# plt.show()
# --------------------------------------

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# print(path_ds)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
# print(image_ds)


# access a few images--------------------------
# plt.figure(10)
# for i,image in enumerate(image_ds.take(4)):
#     plt.subplot(2,2,i+1)
#     plt.imshow(image)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(label_name[all_image_labels[i]])
# plt.show()
# ----------------------------------------------------

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,tf.int64))
# for label in label_ds.take(10):
#     print(label_name[label])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# print(image_label_ds)

# 将数据打乱
BATCH_SIZE =32
# ------------1
# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# ds = ds.prefetch(buffer_size=AUTOTUNE)
# ----------------2
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
# print(ds)

mobile_net = keras.applications.MobileNetV2(input_shape=(180,180,3),include_top=False)
mobile_net.trainable = False
# help(keras.applications.MobileNetV2)

def change_range(image,label):
    return 2*image-1,label
keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
# print(feature_map_batch.shape)




# -----------------------------------------------------------
model = keras.Sequential([
    mobile_net,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(len(label_name),activation='softmax')
])
logit_batch = np.array(model(image_batch))
# steps_per_epoch = np.array(tf.math.ceil(len(all_image_paths)/BATCH_SIZE))

# print("min logit:", np.min(logit_batch))
# print("max logit:", np.max(logit_batch))
# print(logit_batch.shape)

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())


model.fit(ds, epochs=1)
model.save(filepath='D:/ML/model/flow_one/flow')
# --------------------------------------------------------------


# 下面部分测试计算时间,还有一个用文件缓存的方法没有使用
# default_timeit_steps = 2*steps_per_epoch+1
# def timeit(ds,steps=default_timeit_steps):
#     overall_start = time.time()
#     it = iter(ds.take(steps+1))
#     next(it)
#
#     start = time.time()
#     for i,(image, labels) in enumerate(it):
#         if i%10 ==0:
#             print(".",end='')
#     end = time.time()
#     duration = end - start
#     print('{} batchs:{} s'.format(steps, duration))
#     print('{:0.5f} Images/s'.format(BATCH_SIZE * steps/duration))
#     print('Total time: {} s'.format(end - overall_start))
#
# ds = image_label_ds.cache()
# ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
#
# print(timeit(ds))
#---------------------------------------------------------------------------
