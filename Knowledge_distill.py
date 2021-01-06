import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# 让无头老师模型预测后进行，把数据进行软标签处理后当作学生模型（学生模型远小于老师模型）的标签
# 作用是让学生模型快速学习到老师模型的知识，进行模型压缩

chickens = r"D:\ML\images\Chickens"
t_model = keras.models.load_model(r"D:\ML\model\teachermodel.h5")# 加载老师模型
# s_model = keras.models.load_model(r"D:\ML\model\stundentmodel.h5")#加载学生模型
# print(s_model.summary())
# print(t_model.summary())

def softmax_T(x, axis=1, T=1.2):
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
    chickens,
    label_mode='categorical',
    validation_split=0.1,
    subset='training',
    seed=101,
    image_size=(32,32),
    color_mode='grayscale',
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    chickens,
    label_mode='categorical',
    validation_split=0.1,
    subset='validation',
    seed=101,
    image_size=(32,32),
    color_mode='grayscale'
)

# 把图片集焊上
for image_batch, label_batch in train_ds:
    break
train_image = image_batch/255.0
# train_label = label_batch
# print(train_image, train_image.shape)
# print(train_label, train_label.shape)

for image_batch, label_batch in train_ds:
    train_image = np.vstack((train_image, image_batch/255.0))

# 测试集
for image_batch, label_batch in val_ds:
    break
test_image = image_batch/255.0
test_label = label_batch

for image_batch, label_batch in train_ds:
    test_image = np.vstack((test_image, image_batch/255.0))
    test_label = np.vstack((test_label, label_batch))

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


# for l in s_model.layers:
#         l.trainable = True#设置学生模型的所有层都可以训练
# model = keras.Model(s_model.input, s_model.output)
model = tf.keras.Sequential(name='small_model')
model.add(keras.layers.Dense(10, input_shape=(32,32,1), activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2, name='logits'))
model.add(keras.layers.Activation('softmax', name='softmax'))
print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(train_image, t_out,batch_size=32, epochs=50,
          validation_data=(test_image, test_label))

# 在模型最后一层添加 Softmax 后保存
new_model = keras.Sequential()
new_model.add(model)
new_model.add(keras.layers.Softmax())

new_model.save(filepath='D:/ML/model/KnowledgeDistill_Chicken_three')


# 这一部分是加载模型后将模型转化为 tflite 模型，可以部署到Android上

# new_model.save('D:/ML/model/KnowledgeDistill_Chicken_four')
#
# print(model.summary())
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# # Save the model.
# with open(file="D:/ML/lite/chickLite_four.tflite",mode= 'wb') as f:
#   f.write(tflite_model)
