import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

chickens = r"D:\ML\images\Chickens"



train_ds = keras.preprocessing.image_dataset_from_directory(
    chickens,
    validation_split=0.1,
    subset='training',
    seed=101,
    image_size=(32,32),
    color_mode='grayscale',
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    chickens,
    validation_split=0.1,
    subset='validation',
    seed=101,
    image_size=(32,32),
    color_mode='grayscale'
)
class_names = train_ds.class_names
class_names = np.array(class_names)

# 把图片集焊上
for image_batch, label_batch in train_ds:
    break
train_iamge = np.array(image_batch)

for image_batch, label_batch in train_ds:
    train_iamge = np.vstack((train_iamge,image_batch))

print(train_iamge.shape)

# label = np.array(label_batch)
t_model = keras.models.load_model(r"D:\ML\model\teachermodel.h5")
# print(t_model.summary())


# t_model = keras.Sequential([
#     t_model,
#     keras.layers.Softmax()
# ])

# print(t_model.summary())
# predicted_batch = t_model.predict(image_batch)
# predicted_id = np.argmax(predicted_batch, axis=-1)
# predicted_label_batch = class_names[predicted_id]
#
# label_id = np.array(label_batch)
# True_label = class_names[label_id]




s_model = keras.models.load_model(r"D:\ML\model\stundentmodel.h5")#加载学生模型
t_out = t_model.predict(train_iamge)
# print(t_out[1])

print(train_iamge.shape,
      t_out.shape)


for l in s_model.layers:
        l.trainable = True#设置学生模型的所有层都可以训练

model = keras.Model(s_model.input, s_model.output)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(train_iamge, t_out, batch_size=32, epochs=5)
