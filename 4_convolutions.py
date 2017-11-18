#
# Deep Learning
# =============
#
# Assignment 4
# ------------
#
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify
# [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.
#
# The goal of this assignment is make the neural network convolutional.


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from time import time
from math import ceil

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


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
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

# out2 = np.zeros((patch_size * depth, patch_size))
# for i in range(depth):
#     for j in range(depth):
#         out2[(5 * i):(5 * (i + 1)), :] = final_weights_conv2[:, :, 0, i]
# np.savetxt('conv2_weights.csv', out2, delimiter=',')

# Plot outputs from convolutional layer.  See how this changes if you add layers!

# ---
# Problem 2
# ---------
#
# Try to get the best performance you can using a convolutional net. Look for example at the classic
# [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.
#
# Added some logging from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py
#
# ---


def train_cnn(batch_size=16, patch_size=[3], depth=[6], pool_size=2, num_hidden=[], padding='SAME', num_steps=501,
              print_divisor=10, learning_rate=0.1, momentum=0, dropout=0, num_labels=10, log=False):
    """
    Convolutional NN wrapper for TensorFlow.
    :param patch_size: How big should the convolutional patches be?
    :param depth: How many channels should be in the convolutional layers?  Should be of the same length as patches.
    :param pool_size: How large should the pooling be for the max pool layers between the convolutional layers?
    :param num_hidden: list of number of hidden nodes.  The last layer is always a fully connected layer with the
        same number of outputs as the target. 
    :param padding: 'SAME' or 'VALID', convolutional option
    :param num_steps: How many steps to run for
    :param print_divisor: How often to print results/write results to tensorboard logs
    :param learning_rate, momentum, dropout: neural network parameters
    :return: test accuracy of the final network, but also prints results and writes to logs
    """
    n_conv_layers = len(depth)
    num_hidden = num_hidden + [num_labels]
    n_fc_layers = len(num_hidden)

    start_time = time()
    logs_path = 'logs2/' + '_'.join([str(key) + ':' + str(value) for key, value in locals().items()])

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        conv_weights = [None] * n_conv_layers
        conv_biases = [None] * n_conv_layers
        for i in range(n_conv_layers):
            in_channels = num_channels if i == 0 else depth[i-1]
            conv_weights[i] = tf.Variable(
                tf.truncated_normal([patch_size[i], patch_size[i], in_channels, depth[i]], stddev=0.1))
            conv_biases[i] = tf.Variable(tf.zeros([depth[i]]))
        reduced_width = image_size
        for i in range(len(depth)):
            if padding == 'SAME':
                reduced_width = int(ceil(float(reduced_width) / pool_size))
            else:
                reduced_width = int(ceil(float(reduced_width - patch_size[i] + 1) / pool_size))

        fc_weights = [None] * n_fc_layers
        fc_biases = [None] * n_fc_layers
        for i in range(n_fc_layers):
            in_nodes = reduced_width * reduced_width * depth[-1] if i == 0 else num_hidden[i-1]
            fc_weights[i] = tf.Variable(tf.truncated_normal([in_nodes, num_hidden[i]], stddev=0.1))
            fc_biases[i] = tf.Variable(tf.constant(1.0, shape=[num_hidden[i]]))

        # Model
        def model(data):
            for i in range(n_conv_layers):
                input = data if i == 0 else hidden
                conv = tf.nn.conv2d(input, conv_weights[i], [1, 1, 1, 1], padding=padding)
                pool = tf.nn.max_pool(conv, [1, pool_size, pool_size, 1], [1, pool_size, pool_size, 1], padding=padding)
                hidden = tf.nn.relu(pool + conv_biases[i])

            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            for i in range(n_fc_layers):
                input = reshape if i == 0 else tf.nn.relu(fc_output)
                if dropout == 0:
                    fc_output = tf.matmul(input, fc_weights[i]) + fc_biases[i]
                else:
                    fc_output = tf.nn.dropout(tf.matmul(input, fc_weights[i]), dropout) + fc_biases[i]
            return fc_output

        # Training computation
        logits = model(tf_train_dataset)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum).minimize(loss)

        # Predictions for the training, validation, and test data
        train_prediction = tf.nn.softmax(model(tf_train_dataset))
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        with tf.name_scope('Accuracy'):
            train_acc = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
            train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float32))
            valid_acc = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(valid_labels, 1))
            valid_acc = tf.reduce_mean(tf.cast(valid_acc, tf.float32))

        # Create a summary to monitor tensors
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("train_accuracy", train_acc)
        tf.summary.scalar("validation_accuracy", valid_acc)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:
        # session.run(init)
        tf.global_variables_initializer().run()

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions, summary = session.run([optimizer, loss, train_prediction, merged_summary_op],
                                                     feed_dict=feed_dict)
            if step % print_divisor == 0:
                # print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
                # print('.', end='')
                if log:
                    summary_writer.add_summary(summary, num_steps)
        test_accuracy = accuracy(test_prediction.eval(), test_labels)
        print('Test accuracy: %.1f%%' % test_accuracy)

    print('Execution time: {}'.format(round((time() - start_time) / 60, 2)))

    return test_accuracy

train_cnn(batch_size=32, num_steps=201, print_divisor=20, learning_rate=0.2)  # 3 minutes, Error: 84/80/88
train_cnn(batch_size=32, num_steps=1001, print_divisor=20, learning_rate=0.1)  # a while, Error: 80ish/81/...
train_cnn(batch_size=32, patch_size=[5], num_steps=201, print_divisor=20, learning_rate=0.2)  # 6 minutes, E: 84/82/88

train_cnn(batch_size=64, patch_size=[5, 5], depth=[6, 12], num_steps=2001, print_divisor=20, learning_rate=0.1,
          num_hidden=[64], log=True)
train_cnn(batch_size=64, patch_size=[3, 3], depth=[3, 6], num_steps=2001, print_divisor=20, learning_rate=0.1,
          num_hidden=[128], log=True)
train_cnn(batch_size=64, patch_size=[5, 5, 5], depth=[6, 12, 24], num_steps=4001, print_divisor=20, learning_rate=0.1,
          num_hidden=[128], log=True)
train_cnn(batch_size=64, patch_size=[5, 5, 5], depth=[6, 12, 24], num_steps=4001, print_divisor=20, learning_rate=0.05,
          num_hidden=[128], momentum=0.1, log=True)
