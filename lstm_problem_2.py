#
# Deep Learning
# =============
#
# Assignment 6
# ------------
#
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model
# over [Text8](http://mattmahoney.net/dc/textdata) data.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data


text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# Utility functions to map characters to vocabulary IDs and back.

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


print(char2id('a'), char2id('z'), char2id(' '))
print(id2char(1), id2char(26), id2char(0))

# Function to generate a training batch for the LSTM model.

batch_size = 64
num_unrollings = 10


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]

# ---
# Problem 2
# ---------
#
# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters
# like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will
# lead to a very sparse representation that is very wasteful computationally.
#
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs
# themselves.
#
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
#
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this
# [article](http://arxiv.org/abs/1409.2329).
#
# ---


def bigram2id(bigram):
    return char2id(bigram[0]) * 27 + char2id(bigram[1])


def id2bigram(dictid):
    char_id_1 = dictid // 27
    char_id_2 = dictid % 27
    return id2char(char_id_1) + id2char(char_id_2)

# import string
# for i in range(1000):
#     test_string = random.choice(string.lowercase) + random.choice(string.lowercase)
#     assert id2bigram(bigram2id(test_string)) == test_string
#     id = int(random.uniform(0, 27*27))
#     assert bigram2id(id2bigram(id)) == id


def get_bigrams_from_string(string):
    return [string[i:(i+2)] for i in range(len(string) - 1)]


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        # self._last_batch_label = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size), dtype=np.int32)
        label = np.zeros(shape=(self._batch_size), dtype=np.int32)
        for b in range(self._batch_size):
            batch[b] = bigram2id(self._text[self._cursor[b]:self._cursor[b] + 2])
            label[b] = char2id(self._text[self._cursor[b] + 2])
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch, label

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = []
        labels = []
        for step in range(self._num_unrollings):
            new_batch, new_label = self._next_batch()
            batches.append(new_batch)
            labels.append(new_label)
        # self._last_batch_label = (batches[-1], labels[-1])
        return batches, labels

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

[id2bigram(x) for x in train_batches.next()[0][0]]
train_batches.next()[0][0]

embedding_size = 32  # Size of the embedding (mapping bigrams to features of dimension 32)
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    embeddings = tf.Variable(tf.truncated_normal([vocabulary_size ** 2, embedding_size], -0.1, 0.1))
    W_input = tf.Variable(tf.truncated_normal([embedding_size, 4 * num_nodes], -0.1, 0.1))
    W_last_output = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
    b = tf.Variable(tf.zeros([1, 4 * num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    W_final = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b_final = tf.Variable(tf.zeros([vocabulary_size]))

    # Definition of the cell computation.
    def lstm_cell(input, last_output, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        gates = tf.matmul(input, W_input) + tf.matmul(last_output, W_last_output) + b
        input_gate, forget_gate, update, output_gate = tf.split(gates, 4, 1)
        state = tf.sigmoid(forget_gate) * state + tf.sigmoid(input_gate) * tf.tanh(update)
        return tf.sigmoid(output_gate) * tf.tanh(state), state

    # Input data.
    train_bigram_ids = [tf.placeholder(tf.int32, [batch_size])] * (num_unrollings)
    train_labels = [tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size])] * (num_unrollings)

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for input_ids in train_bigram_ids:
        input = tf.nn.embedding_lookup(embeddings, input_ids)
        output, state = lstm_cell(input, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), W_final, b_final)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.concat(train_labels, 0), logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input_bigram_id = tf.placeholder(tf.int32, shape=[1])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                  saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(tf.nn.embedding_lookup(embeddings, sample_input_bigram_id),
                                            saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, W_final, b_final))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches, labels = train_batches.next()
        feed_dict = {}
        for i in range(len(batches)):
            feed_dict[train_bigram_ids[i]] = batches[i]
            dummified_labels = np.zeros((labels[i].shape[0], vocabulary_size))
            for index in range(dummified_labels.shape[0]):
                dummified_labels[index, labels[i][index]] = 1
            feed_dict[train_labels[i]] = dummified_labels
        _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(labels))
            dummified_labels = np.zeros((labels.shape[0], vocabulary_size))
            for index in range(dummified_labels.shape[0]):
                dummified_labels[index, labels[index]] = 1
            print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, dummified_labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    sentence = characters(sample(random_distribution()))[0] +\
                               characters(sample(random_distribution()))[0]
                    feed = [bigram2id(sentence)]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input_bigram_id: feed})
                        next_letter = sample(prediction)
                        sentence += characters(next_letter)[0]
                        feed = [bigram2id(sentence[-2:])]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            # for _ in range(valid_size):
            #     valid_batch, valid_labels = valid_batches.next()
            #     predictions = sample_prediction.eval({sample_input_bigram_id: valid_batch[0]})
            #     valid_labels = np.concatenate(list(valid_labels))
            #     dummified_labels = np.zeros((valid_labels.shape[0], vocabulary_size))
            #     for index in range(dummified_labels.shape[0]):
            #         dummified_labels[index, valid_labels[index]] = 1
            #     valid_logprob = valid_logprob + logprob(predictions, dummified_labels)
            # print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

