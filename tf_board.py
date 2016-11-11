import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.histogram_summary(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    input_holder = tf.placeholder(tf.float32, [None, 1], name='input_holder')
    output_holder = tf.placeholder(tf.float32, [None, 1], name='output_holder')

# add hidden layer
l1 = add_layer(input_holder, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(output_holder - prediction),
                                        reduction_indices=[1]))
    tf.scalar_summary('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

merged = tf.merge_all_summaries()
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('logs', sess.graph)
    sess.run(tf.initialize_all_variables())

    for i in range(1000):
        sess.run(train_step, feed_dict={input_holder: x_data,
                                        output_holder: y_data})
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={input_holder: x_data,
                                                 output_holder: y_data})
            writer.add_summary(result, i)
    print('finished !')


def load_tensorboard(_dir):
    """ os.system("chromium-browser 0.0.0.0:6006") """
    import os

    command = "tensorboard --logdir='%s'" % _dir
    os.system(command)
    resp0 = os.popen(command).readlines()
    infor = resp0[0].split(',')[0]
    if 'Tried to connect to port 6006' == infor:
        resp1 = os.popen('lsof -i:6006').readlines()
        pid = int(resp1[1].split(' ')[1])
        os.system('kill %i' % pid)

    os.system(command)
    return 0





