import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

dataSet = r"D:\ML\images\leaf_image_my\leaf_image_0"
t_model = keras.models.load_model(r"D:\ML\model\leaf_VGG16_four")# 加载老师模型
# s_model = keras.models.load_model(r"D:\ML\model\stundentmodel.h5")#加载学生模型
# new_model = keras.models.load_model(filepath='D:/ML/model/KnowledgeDistill_Chicken_three')
# print(s_model.summary())
print(t_model.summary())
# print(new_model.summary())

IMAGE_SIZE = (150, 150)

def softmax_T(x, axis=1, T=1.5):
    # T 为温度系数
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x/T)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


# 数据处理部分
train_ds = keras.preprocessing.image_dataset_from_directory(
    dataSet,
    label_mode='categorical',
    # validation_split=0.1,
    # subset='training',
    seed=101,
    image_size=IMAGE_SIZE
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    dataSet,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=101,
    image_size=IMAGE_SIZE
)

# 把图片集焊上
for image_batch, label_batch in train_ds:
    break
for image_batch, label_batch in val_ds:
    break

train_image = image_batch/255.0
test_image = image_batch/255.0
test_label = label_batch
# train_label = label_batch
# print(train_image, train_image.shape)
# print(train_label, train_label.shape)

num = 0
for image_batch, label_batch in train_ds:
    train_image = np.vstack((train_image, image_batch/255.0))

    num += 1
    print(num)
    # if num==0:
    #     break

# 测试集

num = 0
for image_batch, label_batch in val_ds:
    test_image = np.vstack((test_image, image_batch/255.0))
    test_label = np.vstack((test_label, label_batch))

    num += 1
    print(num)
    # if num == 0:
    #     break


t_out = t_model.predict(train_image)
# 软标签处理
t_out = softmax_T(t_out)

# print("do you want get scope")
# while input()=='y':
#     print("input your scope\na:")
#     a = int(input())
#     print("b:")
#     b = int(input())
#     print(t_out[a:b])
#     print("do you want get scope")
# ====================

print("train:", train_image.shape,
      "\ntest:", test_image.shape, test_label.shape,
      "\nt_out:", t_out.shape)

model = keras.Sequential(name='leaf_smallModel')
model.add(keras.layers.InputLayer(input_shape=IMAGE_SIZE+(3,)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(16 , name='logits'))
model.add(keras.layers.Softmax())

print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

# model.fit(train_ds, batch_size=32, epochs=50,
#           validation_data=val_ds)
model.fit(train_image, t_out,batch_size=32, epochs=30,
          validation_data=(test_image, test_label))

model.save(filepath='D:/ML/model/KnowledgeDistill_leaf_one')

# new_model = keras.Sequential()
# new_model.add(model)
# new_model.add(keras.layers.Softmax())
#
# new_model.save(filepath='D:/ML/model/KnowledgeDistill_leaf_one')
