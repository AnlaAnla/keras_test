import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import os

FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join('../../.keras/datasets/three_poems',file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_label_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

# for ex in all_labeled_data.take(5):
#     print(ex)

# 这里和教程代码有些区别
tokenizer = tfds.features.text_feature.text_lib.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
# print(vocab_size)

encoder = tfds.features.text_feature.text_lib.TokenTextEncoder(vocabulary_set)
example_text = next(iter(all_labeled_data))[0].numpy()
# print(example_text)
encoded_example = encoder.encode(example_text)
# print(encoded_example)

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encoded_map_fn(text, label):
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

all_encoded_data = all_labeled_data.map(encoded_map_fn)


train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)
test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(test_data))
# print(sample_text[0], sample_labels[0])

vocab_size += 1


model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)
print("\nEval loss: {}, Eval accuracy: {}".format(eval_loss, eval_acc))
