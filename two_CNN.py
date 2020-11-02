import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow import keras
from PIL import Image

dataset_url = '../../.keras/datasets/flower_photos'
data_dir = pathlib.Path(dataset_url)
image_count = len(list(data_dir.glob('*/*.jpg')))

# label_name =  sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
# label_index = dict((name, label) for name,label in enumerate(label_name))
#
# print(image_count)
# print(label_name)
# print(label_index)


# roses = list(data_dir.glob('roses/*'))
# tulips = list(data_dir.glob('tulips/*'))

Batch_size = 32
img_heigh = 180
img_width = 180

# keras 加载图片文件
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_heigh, img_width),
    batch_size=Batch_size
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_heigh, img_width),
    batch_size=Batch_size
)
# 读取文件夹名称
class_name = train_ds.class_names
# ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# print(class_name)

# 加载查看几张图片
# plt.figure(figsize=(10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_name[labels[i]])
#         plt.axis('off')
# plt.show()

# 查看一批数据的结构
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break

# 预处理
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# normalixation_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
#
# normalized_ds = train_ds.map(lambda x, y:(normalixation_layer(x),y))
# image_batch, label_batch = next(iter(normalized_ds))
# first_image = image_batch[0]




'''
# 图片随机变换
data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip('horizontal',
                                                       input_shape=(img_heigh,
                                                                    img_width,
                                                                    3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

# plt.figure(figsize=(10,10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         aug_images = data_augmentation(images)
#         plt.subplot(3,3,i+1)
#         plt.imshow(aug_images[0].numpy().astype('uint8'))
#         plt.axis('off')
# plt.show()


# 模型
num_classes = 5
model = keras.Sequential([
    data_augmentation,
    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32, 3,padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(244,activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

print(model.summary())

epochs = 18
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

model.save(filepath='D:/ML/model/flow_two_byCNN')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(121)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validition Accuracy')

plt.subplot(122)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

'''
img_path = r"C:\Users\伊尔安拉\Pictures\Camera Roll\cereals-100263_1920.jpg"
model = keras.models.load_model(filepath='D:/ML/model/flow_two_byCNN')
# print(model.summary())
image = keras.preprocessing.image.load_img(
    path= img_path,
    target_size=(180,180)
)
image_array = keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array,0)

preditons = model.predict(image_array)
score = tf.nn.softmax(preditons)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_name[np.argmax(score)], 100 * np.max(score))
)

plt.imshow(image)
plt.show()
