import tensorflow as tf
from  tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = list(range(10))
train_images = train_images/255.0
test_images = test_images/255.0

img = Image.open('D:/ML/images/th2.png')
img = Image.Image.resize(img,(28,28))
img = img.convert('L')
img = ImageOps.invert(img)
img = np.array(img)/255.0
plt.imshow(img,cmap='binary')
plt.show()

img = np.expand_dims(img,0)
# print(img.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('------\ntest accuracy:',test_acc)
#
probability_model = keras.Sequential([model,
                                      keras.layers.Softmax()])
predictions = probability_model.predict(test_images)



predict_1 = probability_model.predict(img)
print(class_names[int(np.argmax(predict_1))])
print(predict_1)
print("\n\n-------\n",img)
