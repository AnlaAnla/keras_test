import tensorflow as tf
import numpy as np
import os
import time

# 记住下面的方法
path_to_fiel = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# print(path_to_fiel)
# 精彩的读取方式
text = open(path_to_fiel, 'rb').read().decode(encoding='utf-8')
# print('Length of text: {} characters'.format(len(text)))

# print(text[:250])

# 精彩，读取非重复字符
vocab = sorted(set(text))
# print(vocab)
# print(len(vocab))

# 创建从非重复字符到索引的映射
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 将文本转化为数字
text_as_int = np.array([char2idx[c] for c in text])

# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('     {:4s}:{:3d},'.format(repr(char), char2idx[char]))
# print('     ...\n')

# 显示文本首 13 个字符的整数映射
# print("{}---- characters mapped to int -----> {}".format(repr(text[:13]), text_as_int[:13]))

seq_length = 100
examples_per_epoch = len(text)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# for i in char_dataset.take(5):
#     print(ind2char[i.numpy()])

sequence = char_dataset.batch(seq_length+1, drop_remainder=True)

# for item in sequence.take(5):
#     print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequence.map(split_input_target)

# 对应查看数据集
# for input_example, target_example in dataset.take(1):
#     print('Input data', repr(''.join(idx2char[input_example.numpy()])))
#     print('Target data', repr(''.join(idx2char[target_example.numpy()])))
#
# for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#     print('Step {:4d}'.format(i))
#     print("  input: {}({:s})".format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BATCH_SIZE).batch(BATCH_SIZE ,drop_remainder=True)
print(dataset)

# 词集的长度
vocab_size = len(vocab)
# 嵌入的长度
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# model = build_model(vocab_size, embedding_dim,
#                     rnn_units, BATCH_SIZE)
# print(model.summary())

# 简单训练模型
# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape)
# # (64, 100, 65) # (batch_size, sequence_length, vocab_size)

# 测试模型的输出
# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
#
# print(sampled_indices)
# print('Input: \n' ,repr(''.join(idx2char[input_example_batch[0]])))
# print()
# print('Next Chae Predictions: \n', repr(''.join(idx2char[sampled_indices])))


# 训练
# def loss(labels, logist):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels, logist, from_logits=True)
#
# example_batch_loss = loss(target_example_batch, example_batch_predictions)
#
# model.compile(optimizer='adam', loss=loss)
#
# checkpoint_dir = r"D:\ML\model\Text_classification\train_checkpoints"
# checkpoints_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoints_prefix,
#     save_weights_only=True
# )
#
# EPOCHS = 10
#
# history = model.fit(dataset, epochs=EPOCHS ,callbacks=[checkpoint_callback])



# ==========================================================


checkpoint_dir = r"D:\ML\model\Text_classification\train_checkpoints"
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string):
    # 评估步骤（用学习过的模型生成文本）

    # 要生成的字符个数
    num_generate = 1000

    #   将起始字符串转换为数字（向量化）
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 空字符串用于存储结果
    text_generated = []

    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过实验找到最好的设定
    temperature = 3.0

    #这里批大小为 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)

        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predictions_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predictions_id], 0)

        text_generated.append(idx2char[predictions_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))

