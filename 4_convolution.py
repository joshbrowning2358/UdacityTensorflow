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
num_channels = 1 # grayscale

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

def fit_model_and_get_test_error(batch_size=16, patch_size=5, depth_1=6, depth_2=16, num_hidden=64, padding='SAME',
                                 num_steps=251, print_divisor=25, learning_rate=0.05, momentum=0.1, dropout=0):
    logs_path = 'logs2/' + '_'.join([str(key) + ':' + str(value) for key, value in locals().items()])

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth_1], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth_1]))
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_1, depth_2], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))
        if padding == 'SAME':
            reduced_width = image_size / 4
        else:
            reduced_width = ((image_size - patch_size + 1) / 2 - patch_size + 1) / 2
        layer3_weights = tf.Variable(
            tf.truncated_normal([reduced_width * reduced_width * depth_2, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model
        def model(data):
            # with tf.name_scope('Conv1'):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding=padding)
            # with tf.name_scope('Pool1'):
            pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding=padding)
            # with tf.name_scope('Hidden1'):
            hidden = tf.nn.relu(pool + layer1_biases)
            # with tf.name_scope('Conv2'):
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding=padding)
            # with tf.name_scope('Pool2'):
            pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding=padding)
            # with tf.name_scope('Hidden2'):
            hidden = tf.nn.relu(pool + layer2_biases)

            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            if dropout == 0:
                # with tf.name_scope('Fully_Connected1'):
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                # with tf.name_scope('Logits'):
                logits = tf.matmul(hidden, layer4_weights) + layer4_biases
            else:
                # with tf.name_scope('Fully_Connected1'):
                hidden = tf.nn.relu(tf.nn.dropout(tf.matmul(reshape, layer3_weights), dropout) + layer3_biases)
                # with tf.name_scope('Logits'):
                logits = tf.nn.dropout(tf.matmul(hidden, layer4_weights), dropout) + layer4_biases
            return logits

        # Training computation
        logits = model(tf_train_dataset)
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # with tf.name_scope('SGD'):
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
                # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                # print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
                print('.', end='')
                summary_writer.add_summary(summary, num_steps)
        test_accuracy = accuracy(test_prediction.eval(), test_labels)
        print('Test accuracy: %.1f%%' % test_accuracy)
        final_weights_conv1 = layer1_weights.eval()
        final_weights_conv2 = layer2_weights.eval()

    return test_accuracy

fit_model_and_get_test_error(dropout=0)  # 90.6
for learning_rate in [0.001, 0.01, 0.1, 1]:
    for momentum in [0.0000000001, 0.01, 0.1, 1]:
        for dropout in [0, 0.1, 0.3]:
            fit_model_and_get_test_error(learning_rate=learning_rate, momentum=momentum, dropout=dropout)

