# ---
#
# Problem
# -------
#
# An alternative to skip-gram is another Word2Vec model called [CBOW](http://arxiv.org/abs/1301.3781) (Continuous Bag of
# Words). In the CBOW model, instead of predicting a context word from a word vector, you predict a word from the sum
# of all the word vectors in its context. Implement and evaluate a CBOW model trained on the text8 dataset.
#
# ---

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data('text8.zip')
print('Data size %d' % len(words))

vocabulary_size = 50000

# Build Dataset
count = [('UNK', -1)]
count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
count[0] = ('UNK', len(words) - sum(x[1] for x in count))  # all other words are unknown: 'UNK'
dictionary = {key: value[0] for key, value in enumerate(count)}
reverse_dictionary = {value: key for key, value in dictionary.items()}
data = [reverse_dictionary.get(word, 0) for word in words]
del words


def generate_batch(batch_size, window_width, start_index=0):
    buffer = collections.deque(maxlen=window_width)
    buffer_center_index = window_width // 2
    index = start_index
    input_data = []
    target = []
    for i in range(window_width - 1):
        buffer.append(data[index])
        index += 1
    for i in range(batch_size):
        buffer.append(data[index])
        index += 1
        if index > len(data):  # reset buffer at start of word list (don't want wrapping).
            index = window_width
            for i in range(window_width):
                buffer.append(data[index])
        current_words = list(buffer)
        target.append([current_words.pop(buffer_center_index)])
        input_data.append(current_words)
    return input_data, target


def build_model(batch_size, window_width, embedding_size, negative_sample, learning_rate, num_steps=100001):
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        input = tf.placeholder(tf.int32, (batch_size, window_width - 1))
        labels = tf.placeholder(tf.int32, (batch_size, 1))

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([vocabulary_size]))

        embed = tf.reduce_mean(tf.nn.embedding_lookup(embeddings, ids=input), axis=1)
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=weights, biases=biases, inputs=embed, labels=labels,
                                                         num_sampled=negative_sample, num_classes=vocabulary_size))

        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        start_index = 0

        for step in range(num_steps):
            input_data, target = generate_batch(batch_size, window_width, start_index=start_index)
            _, l = session.run([optimizer, loss], feed_dict={input: input_data, labels: target})
            start_index += batch_size

            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
        raw_embedding = embeddings.eval()
        final_embedding = normalize(raw_embedding)

    return final_embedding

embedding = build_model(batch_size=128, window_width=5, embedding_size=128, negative_sample=30, learning_rate=0.1,
                        num_steps=100001)

dist = np.dot(embedding, np.transpose(embedding[reverse_dictionary['small']]))
sorted_ids = sorted(range(50000), key=lambda x: dist[x])
[dictionary[x] for x in sorted_ids[49990:]]