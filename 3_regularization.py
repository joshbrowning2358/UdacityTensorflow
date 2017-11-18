# Deep Learning
#
# Assignment 3
#
# Previously in 2_fullyconnected.ipynb, you trained a logistic regression and a neural network model.
#
# The goal of this assignment is to explore regularization techniques.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from copy import copy

# First reload the data we generated in `1_notmnist.ipynb`.

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

# ---
# Problem 1
# ---------
#
# Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding
# a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using
# `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.
#
# ---

batch_size = 128
num_steps = 5000
hidden_layers = 1028
learning_rate = 0.5
l2_constant = 0.001

# graph = tf.Graph()
# with graph.as_default():
#     train = tf.placeholder(tf.float32, [batch_size, image_size * image_size])
#     labels = tf.placeholder(tf.float32, [batch_size, num_labels])
#     valid = tf.constant(valid_dataset)
#     test = tf.constant(test_dataset)
#
#     W = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels], mean=0, stddev=0.5))
#     bias = tf.Variable(tf.zeros([num_labels, ]))
#
#     logits = tf.matmul(train, W) + bias
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) +\
#         l2_constant * tf.nn.l2_loss(W)
#
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
#     prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(tf.matmul(valid, W) + bias)
#     test_prediction = tf.nn.softmax(tf.matmul(test, W) + bias)

graph = tf.Graph()
with graph.as_default():
    train = tf.placeholder(tf.float32, [batch_size, image_size * image_size])
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    valid = tf.constant(valid_dataset)
    test = tf.constant(test_dataset)

    W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layers], mean=0, stddev=0.5))
    bias1 = tf.Variable(tf.zeros([hidden_layers, ]))
    W2 = tf.Variable(tf.truncated_normal([hidden_layers, num_labels], mean=0, stddev=0.5))
    bias2 = tf.Variable(tf.zeros([num_labels, ]))

    layer1_out = tf.nn.relu(tf.matmul(train, W1) + bias1)
    logits = tf.matmul(layer1_out, W2) + bias2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) +\
        l2_constant * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    prediction = tf.nn.softmax(logits)
    valid_hidden = tf.nn.relu(tf.matmul(valid, W1) + bias1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_hidden, W2) + bias2)
    test_hidden = tf.nn.relu(tf.matmul(test, W1) + bias1)
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden, W2) + bias2)

# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     for step in range(num_steps):
#         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#
#         _, l, preds = session.run([optimizer, loss, prediction], feed_dict={train: batch_data, labels: batch_labels})
#         # _, loss = session.run([optimizer, loss], feed_dict={train: batch_data, labels: batch_labels})
#         if step % (num_steps/20) == 0:
#             print('Minibatch loss: {}'.format(l))
#             print('Minibatch valid accuracy: {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
#     print('Test accuracy: {}'.format(accuracy(test_prediction.eval(), test_labels)))


# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
#
# ---

batch_size = 128
num_steps = 500
hidden_layers = 1028
learning_rate = 0.5
l2_constant = 0.001

graph = tf.Graph()
with graph.as_default():
    train = tf.placeholder(tf.float32, [batch_size, image_size * image_size])
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    valid = tf.constant(valid_dataset)
    test = tf.constant(test_dataset)

    W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layers], mean=0, stddev=0.5))
    bias1 = tf.Variable(tf.zeros([hidden_layers, ]))
    W2 = tf.Variable(tf.truncated_normal([hidden_layers, num_labels], mean=0, stddev=0.5))
    bias2 = tf.Variable(tf.zeros([num_labels, ]))

    layer1_out = tf.nn.relu(tf.matmul(train, W1) + bias1)
    logits = tf.matmul(layer1_out, W2) + bias2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) +\
        l2_constant * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    prediction = tf.nn.softmax(logits)
    valid_hidden = tf.nn.relu(tf.matmul(valid, W1) + bias1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_hidden, W2) + bias2)
    test_hidden = tf.nn.relu(tf.matmul(test, W1) + bias1)
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden, W2) + bias2)

# with tf.Session(graph=graph) as session:
#     tf.global_variables_initializer().run()
#     for step in range(num_steps):
#         offset = 128 * (num_steps % 4)
#
#         batch_data = train_dataset[offset:(offset + batch_size), :]
#         batch_labels = train_labels[offset:(offset + batch_size), :]
#
#         _, l, preds = session.run([optimizer, loss, prediction], feed_dict={train: batch_data, labels: batch_labels})
#         # _, loss = session.run([optimizer, loss], feed_dict={train: batch_data, labels: batch_labels})
#         if step % (num_steps/20) == 0:
#             print('Minibatch loss: {}'.format(l))
#             print('Minibatch accuracy: {}'.format(accuracy(preds, batch_labels)))
#             print('Minibatch valid accuracy: {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
#     print('Test accuracy: {}'.format(accuracy(test_prediction.eval(), test_labels)))



# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during
# training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides
# `nn.dropout()` for that, but you have to make sure it's only inserted during training.
#
# What happens to our extreme overfitting case?
#
# ---

batch_size = 128
num_steps = 5000
hidden_layers = 1028
learning_rate = 0.5
l2_constant = 0.001
drop_prob = 0.9

import time
start = time.time()

graph = tf.Graph()
with graph.as_default():
    train = tf.placeholder(tf.float32, [batch_size, image_size * image_size])
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    valid = tf.constant(valid_dataset)
    test = tf.constant(test_dataset)

    W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layers], mean=0, stddev=0.5))
    bias1 = tf.Variable(tf.zeros([hidden_layers, ]))
    W2 = tf.Variable(tf.truncated_normal([hidden_layers, num_labels], mean=0, stddev=0.5))
    bias2 = tf.Variable(tf.zeros([num_labels, ]))

    layer1_out = tf.nn.relu(tf.nn.dropout(tf.matmul(train, W1), drop_prob) + bias1)
    logits = tf.nn.dropout(tf.matmul(layer1_out, W2), drop_prob) + bias2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)) +\
        l2_constant * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    batch_hidden = tf.nn.relu(tf.matmul(train, W1) + bias1)
    batch_prediction = tf.nn.softmax(tf.matmul(batch_hidden, W2) + bias2)
    valid_hidden = tf.nn.relu(tf.matmul(valid, W1) + bias1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_hidden, W2) + bias2)
    test_hidden = tf.nn.relu(tf.matmul(test, W1) + bias1)
    test_prediction = tf.nn.softmax(tf.matmul(test_hidden, W2) + bias2)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        _, l = session.run([optimizer, loss], feed_dict={train: batch_data, labels: batch_labels})
        if step % (num_steps/10) == 0:
            print('Minibatch loss: {}'.format(l))
            print('Validation accuracy: {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
    print('Test accuracy: {}'.format(accuracy(test_prediction.eval(), test_labels)))

print('Processed in {} minutes'.format(round((time.time() - start)/60, 2)))

# 94.05 test:
batch_size = 128
num_steps = 5000
hidden_layers = 1028
learning_rate = 0.5
l2_constant = 0.001
drop_prob = 0.9


# ---
# Problem 4
# ---------
#
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep
# network is [97.1%]
# (http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
#
# One avenue you can explore is to add multiple layers.
#
# Another one is to use learning rate decay:
#
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#
#  ---


batch_size = 128
num_steps = 5000
hidden_layers = [256]
learning_rate = 0.5
l2_constant = 0.001
drop_prob = 0.9


def evaluate_network(hidden_layers, batch_size=128, num_steps=5000, learning_rate=0.5, l2_constant=0.001, drop_prob=0.9,
                     store_results=20):
    batch_error = []
    valid_error = []
    graph = tf.Graph()
    with graph.as_default():
        train = tf.placeholder(tf.float32, [batch_size, image_size * image_size])
        train_layers = [train]
        labels = tf.placeholder(tf.float32, [batch_size, num_labels])
        valid = tf.constant(valid_dataset)
        test = tf.constant(test_dataset)

        weight_sizes = [image_size * image_size] + hidden_layers + [num_labels]
        W = [tf.Variable(tf.truncated_normal([size_in, size_out], mean=0, stddev=0.5))
             for size_in, size_out in zip(weight_sizes[:-1], weight_sizes[1:])]
        bias_sizes = hidden_layers + [num_labels]
        bias = [tf.Variable(tf.zeros([size, ])) for size in bias_sizes]

        valid_layer = valid
        test_layer = test
        for i in range(len(W)):
            if i < (len(W) - 1):
                train_layer = tf.nn.relu(tf.nn.dropout(tf.matmul(train_layers[i], W[i]), drop_prob) + bias[i])
                train_layers = train_layers + [train_layer]
                valid_layer = tf.nn.relu(tf.matmul(valid_layer, W[i]) + bias[i])
                test_layer = tf.nn.relu(tf.matmul(test_layer, W[i]) + bias[i])
            else:
                train_layer = tf.nn.dropout(tf.matmul(train_layers[i], W[i]), drop_prob) + bias[i]
                train_layers = train_layers + [train_layer]
                valid_layer = tf.matmul(valid_layer, W[i]) + bias[i]
                test_layer = tf.matmul(test_layer, W[i]) + bias[i]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=train_layers[-1])) +\
            l2_constant * sum(tf.nn.l2_loss(W_i) for W_i in W)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        batch_prediction = tf.nn.softmax(train_layers[-1])
        valid_prediction = tf.nn.softmax(valid_layer)
        test_prediction = tf.nn.softmax(test_layer)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            _, l, preds = session.run([optimizer, loss, batch_prediction],
                                      feed_dict={train: batch_data, labels: batch_labels})
            if step % (num_steps/store_results) == 0:
                # print('Minibatch loss: {}'.format(l))
                # print('Validation accuracy: {}'.format(accuracy(valid_prediction.eval(), valid_labels)))
                batch_error += [accuracy(preds, batch_labels)]
                valid_error += [accuracy(valid_prediction.eval(), valid_labels)]
        test_accuracy = accuracy(test_prediction.eval(), test_labels)
        print('Test accuracy: {}'.format(test_accuracy))

    return batch_error, valid_error, test_accuracy


# Test Loss: 94.4
batch_size = 128; num_steps = 5000; hidden_layers = [1028]; learning_rate = 0.5; l2_constant = 0.001; drop_prob = 0.9

layers = [
    [128], [256], [512], [1024], [2048],
    [128, 128], [128*2, 128], [128*3, 128],
    [256, 256], [256*2, 256], [256*3, 256],
    [512, 512], [512 * 2, 512], [512 * 3, 512],
    [2048, 1024, 512, 256]
]

params = [{'hidden_layers': l, 'learning_rate': 0.5 / (2**(len(l) - 1))} for l in layers]
smaller_lr = [{'hidden_layers': x['hidden_layers'], 'learning_rate': x['learning_rate'] / 5} for x in params]
params = params + smaller_lr

test_error = map(lambda x: evaluate_network(**x), params)

[x[2] for x in test_error]