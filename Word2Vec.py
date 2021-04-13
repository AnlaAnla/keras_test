import io
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

# sentence = "The wide road shimmered in the hot sun"
# tokens = list(sentence.lower().split())
# # print(len(tokens))
#
# vocab, index = {}, 1
# vocab['<pad>'] = 0
# for token in tokens:
#     if token not in vocab:
#         vocab[token] = index
#         index += 1
# # creat a vocabulary to save mappings from tokens to integer indices
# vocab_size = len(vocab)
# # print(vocab)
#
# # creat an inverse vocabulary to save mapping from integer indices to tokens
# inverse_vocab = {index: token for token, index in vocab.items()}
# # print(inverse_vocab)
#
# example_sequence = [vocab[word] for word in tokens]
# # print(example_sequence)
#
#
# window_size = 2
# positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
#     example_sequence,
#     vocabulary_size=vocab_size,
#     window_size = window_size,
#     negative_samples=0
# )
# # print(len(positive_skip_grams))
#
# # for target, context in positive_skip_grams[:5]:
# #     print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
#
# target_word, context_word = positive_skip_grams[0]
num_ns = 4
#
# context_class = tf.reshape(tf.constant(context_word, dtype='int64'), (1, 1))
# negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
#     true_classes=context_class,
#     num_true=1,
#     num_sampled=num_ns,
#     unique=True,
#     range_max=vocab_size,
#     seed=SEED,
#     name="negative_sampling"
# )
# # print(negative_sampling_candidates)
# # print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])
#
# negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
# context = tf.concat([context_class, negative_sampling_candidates], 0)
# label = tf.constant([1] + [0]*num_ns, dtype="int64")
#
# target = tf.squeeze(target_word)
# context = tf.squeeze(context)
# label = tf.squeeze(label)
#
# # print(f"target_index    : {target}")
# # print(f"target_word     : {inverse_vocab[target_word]}")
# # print(f"context_indices : {context}")
# # print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
# # print(f"label           : {label}")
# # print("target  :", target)
# # print("context :", context)
# # print("label   :", label)
#
# sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
# print(sampling_table)


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
with open(path_to_file) as f:
    lines = f.read().splitlines()
# for line in lines[:20]:
#     print(line)

text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                   '[%s]'%re.escape(string.punctuation), '')

vocab_size = 4096
sequence_length = 10

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocabe = vectorize_layer.get_vocabulary()
# print(inverse_vocabe)

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
# print(len(sequences))

# for seq in sequences[:5]:
#     print(f"{seq} => {[inverse_vocabe[i] for i in seq]}")

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED
)
# print(len(targets), len(contexts), len(labels))

BATCH_SEZE = 1024
BUFFER_SEZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SEZE).batch(BUFFER_SEZE, drop_remainder=True)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
# print(dataset)

class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3, 2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
