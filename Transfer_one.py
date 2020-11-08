import numpy as np
import os
import time
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

IMAGE_SHAPE = (224,224)
# classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#
# classifier = keras.Sequential([
#     hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
# ])

grace_hopper = Image.open(r"D:\ML\images\可爱\1\夏天 鲜花草地 可爱娃娃女孩桌面壁纸.jpg").resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0

# result = classifier.predict(grace_hopper[np.newaxis, ...])
# print(result.shape)

# prediction = np.argmax(result[0],axis=-1)
# print(prediction)

# labels_path = keras.utils.get_file('ImageNetLabels.txt', 'http://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# image_label = np.array(open(labels_path).read().splitlines())

# 使用模型预测一波
# plt.imshow(grace_hopper)
# plt.axis('off')
# predicted_class_name = image_label[prediction]
# plt.title('prediction '+ predicted_class_name.title())
# plt.show()

data_root = tf.keras.utils.get_file('flower_photos',
                                    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                    untar=True)

image_generation = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generation.flow_from_directory(str(data_root),target_size=IMAGE_SHAPE)

for image_batch, labels_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", labels_batch.shape)
    break

# result_batch = classifier.predict(image_batch)
# predicted_class_names = image_label[np.argmax(result_batch, axis=-1)]

# print(predicted_class_names)

# plt.figure(figsize=(10,10))
# plt.subplots_adjust(hspace=0.5)
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.imshow(image_batch[n])
#     plt.title(predicted_class_names[n])
#     plt.axis('off')
# plt.suptitle("ImageNet prediction")
# plt.show()


# 加载无头模型用于迁移学习
feature_extractor_layer = hub.KerasLayer(r"D:\ML\transfer_hub\classifier",input_shape=(224,224,3),trainable=False)

model = keras.Sequential([
    feature_extractor_layer,
    keras.layers.Dense(image_data.num_classes)
])

model.compile(
    optimizer= keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[batch_stats_callback])

model.save(filepath='D:/ML/model/transfer_one')

# 作图
plt.figure()

plt.subplot(211)
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.subplot(212)
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

plt.show()







# ============================================================================
# ==================================== 下面代码独立进行，加载模型后预测

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = (224,224)

image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(directory='../../.keras/datasets/flower_photos',target_size=IMAGE_SIZE)

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])

for image_batch, label_batch in image_data:
    break
# print(image_batch.shape)

# print(class_names)

model = keras.models.load_model(filepath='D:/ML/model/transfer_one')

model = keras.Sequential([
    model,
    keras.layers.Softmax()
])

print(model.summary())
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch,axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)
True_label = class_names[label_id]

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = 'green' if predicted_id[n] == label_id[n] else 'red'
    plt.title("pre{},Tr({})".format(predicted_label_batch[n].title(),True_label[n].title()), color=color)
    plt.axis('off')
plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()
