from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)

input_holder = tf.placeholder("float", [None, n_steps, n_input])
label_holder = tf.placeholder("float", [None, n_classes])
state_holder = tf.placeholder("float", [None, 2 * n_hidden])  # LSTM cell requires 2x n_hidden length (state & cell)

# Define weights
weights = {'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
           'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}

biases = {'hidden': tf.Variable(tf.random_normal([n_hidden])),
          'out': tf.Variable(tf.random_normal([n_classes]))}


_X = tf.transpose(input_holder, [1, 0, 2])  # permute n_steps and batch_size
_X = tf.reshape(_X, [-1, n_input])  # (n_steps*batch_size, n_input)
_X = tf.matmul(_X, weights['hidden']) + biases['hidden']
_X = tf.split(0, n_steps, _X)  # n_steps * (batch_size, n_hidden)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=False)
outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=state_holder)
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']  # Get inner loop last output


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label_holder))  # Softmax loss
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)  # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch_size = 128

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < 100000:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
        # Fit training using batch data
        sess.run(train_op, feed_dict={input_holder: batch_xs,
                                      label_holder: batch_ys,
                                      state_holder: np.zeros((batch_size, 2 * n_hidden))})
        if step % 100 == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={input_holder: batch_xs,
                                                label_holder: batch_ys,
                                                state_holder: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={input_holder: batch_xs,
                                             label_holder: batch_ys,
                                             state_holder: np.zeros((batch_size, 2 * n_hidden))})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                  ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Finished!")
    # Calculate accuracy for 256 mnist test images
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Accuracy:", sess.run(accuracy, feed_dict={input_holder: test_data,
                                                     label_holder: test_label,
                                                     state_holder: np.zeros((test_len, 2 * n_hidden))}))
