import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from keras.applications.vgg16 import VGG16

IMAGE_SIZE = (150, 150)
model_path = r"D:\ML\model\leaf_VGG16_three"
Image_train = r"D:\ML\images\leaf_image_my\leaf_image_0"
Image_test = r"D:\ML\images\leaf_image_my\leaf_image_test"
epochs = 200

image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                               # rotation_range=40,
                                                               # width_shift_range=0.2,
                                                               # height_shift_range=0.2,
                                                               # shear_range=0.2,
                                                               # zoom_range=0.2,
                                                               # horizontal_flip=True,
                                                               # fill_mode='nearest'
                                                               )

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/255
    )

image_data = image_generator.flow_from_directory(directory=Image_train, target_size=IMAGE_SIZE)
test_generator = test_datagen.flow_from_directory(Image_test, target_size=IMAGE_SIZE)

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)


classes = image_data.num_classes
# 构建模型
# model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_SIZE+(3,))
# for layer in model_vgg.layers:
#     layer.trainable = False
#
# model = layers.Flatten(name='flatten')(model_vgg.output)  # 去掉全连接层，前面都是卷积层
# model = layers.Dense(4096, activation='relu', name='fc1')(model)
# model = layers.Dense(4096, activation='relu', name='fc2')(model)
# model = layers.Dropout(0.5)(model)
# model = layers.Dense(classes, activation='softmax')(model)  # model就是最后的y

# model_vgg = keras.Model(inputs=model_vgg.input, outputs=model, name='vgg16')





# 加载模型并设置训练层
model = keras.models.load_model(model_path)

for layer in model.layers:
    layer.trainable = False
# 或者使用如下方法冻结所有层
# model.trainable = False
model.layers[-1].trainable = True


for x in model.trainable_weights:
    print(x.name)
print('===========================\n')

# 不可训练层
for x in model.non_trainable_weights:
    print(x.name)
print('\n')


model_vgg = model

# 打印模型结构，包括所需要的参数
model_vgg.summary()


model_vgg.compile(
    optimizer= keras.optimizers.SGD(learning_rate=0.005, decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model_vgg.fit_generator(image_data,
                              steps_per_epoch=len(image_data),
                              epochs=epochs,
                              validation_data=test_generator,
                              validation_steps=len(test_generator))

# 保存模型
model_vgg.save(filepath='D:/ML/model/leaf_VGG16_four')
print(history)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validition Accuracy')
plt.subplot(212)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()



# import numpy as np
# import os
# import time
# import PIL.Image as Image
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow import keras
#
# os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
#
# IMAGE_SIZE = (224, 224)
# Image_train = r"D:\ML\images\leaf_image_my\leaf_image_0"
# Image_test = r"D:\ML\images\leaf_image_my\leaf_image_test"
# epochs = 5
#
# train_datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
#                                                                width_shift_range=0.2,
#                                                                height_shift_range=0.2,
#                                                                rescale=1/255,
#                                                                shear_range=0.2,
#                                                                zoom_range=0.2,
#                                                                horizontal_flip=True,
#                                                                fill_mode='nearest'
#                                                                )
#
# test_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale = 1/255
#     )
#
# train_generator = train_datagen.flow_from_directory(directory=Image_train, target_size=IMAGE_SIZE)
# test_generator = test_datagen.flow_from_directory(Image_test, target_size=IMAGE_SIZE)
#
# # for image_batch, labels_batch in image_data:
# #     print("Image batch shape: ", image_batch.shape)
# #     print("Label batch shape: ", labels_batch.shape)
# #     break
#
# # 加载无头模型用于迁移学习
# feature_extractor_layer = hub.KerasLayer(r"D:\ML\tfhub_modules\imagenet_mobilenet_v2_100_224_feature_vector_4",input_shape=IMAGE_SIZE+(3,), trainable=False)
#
# model = keras.Sequential([
#     feature_extractor_layer,
#     keras.layers.Dense(train_generator.num_classes)
# ])
#
# model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=len(train_generator),
#                               epochs=20,
#                               validation_data=test_generator,
#                               validation_steps=len(test_generator))
#
# # 保存模型
# model.save(filepath='D:/ML/model/leaf_VGG16_one')
#
#
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(epochs)
# plt.figure(figsize=(8,8))
# plt.subplot(121)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend()
# plt.title('Training and Validition Accuracy')
# plt.subplot(122)
# plt.plot(epochs_range, loss, label="Train Loss")
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend()
# plt.title("Training and Validation Loss")
# plt.show()