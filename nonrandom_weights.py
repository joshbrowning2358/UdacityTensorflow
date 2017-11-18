import tensorflow as tf
import numpy as np

n_features = 10
batch_size = 128
n_hidden = 10
learning_rate = 0.01
num_steps = 1000

train = np.random.random((10000, n_features))
target = train.sum(axis=1) + np.random.random(10000) * 0.1

graph = tf.Graph()

with graph.as_default():
    tf_train = tf.placeholder(tf.float32, (batch_size, n_features))
    tf_target = tf.placeholder(tf.float32, (batch_size, ))

    # hidden_weights = tf.Variable(tf.truncated_normal(shape=(n_features, n_hidden), mean=1/n_features, stddev=0.1))
    hidden_weights = tf.Variable(tf.zeros(shape=(n_features, n_hidden)))
    hidden_biases = tf.Variable(tf.zeros(shape=(n_hidden, )))
    output_weights = tf.Variable(tf.truncated_normal(shape=(n_hidden, 1), mean=1/n_hidden, stddev=0.1))
    # output_weights = tf.Variable(tf.zeros(shape=(n_hidden, 1)))
    output_bias = tf.Variable(tf.zeros(shape=(1, )))

    hidden_layer = tf.matmul(tf_train, hidden_weights) + hidden_biases
    estimate = tf.matmul(hidden_layer, output_weights)

    loss = tf.reduce_mean((tf_target - estimate) * (tf_target - estimate))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (target.shape[0] - batch_size)

        batch_data = train[offset:(offset + batch_size), :]
        batch_target = target[offset:(offset + batch_size),]

        _, l = session.run([optimizer, loss], feed_dict={tf_train: batch_data, tf_target: batch_target})
        if step % 100 == 0:
            print('RMSE : {}'.format(l))
    print(hidden_weights.eval())
    print(output_weights.eval())

# Are all weights the same if initialized to 0?
# What if dropout is present?
# Confirm they don't match when randomly initialized