import tensorflow as tf
from collections import namedtuple
from tensorflow.examples.tutorials.mnist import input_data
import time


"""Test the resnet on MNIST."""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

dim_img_x = 28
dim_in = dim_img_x * dim_img_x
dim_out = 10
X_holder = tf.placeholder(tf.float32, [None, dim_in])
Y_holder = tf.placeholder(tf.float32, [None, dim_out])

'''||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'''
'''' ...... drnn begin '''

LayerBlock = namedtuple('LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
blocks = [LayerBlock(3, 128, 32), LayerBlock(3, 256, 64), LayerBlock(3, 512, 128), LayerBlock(3, 1024, 256)]

# First convolution expands to 64 channels and downsamples
I_1 = tf.reshape(X_holder, [-1, dim_img_x, dim_img_x, 1])

W_1 = tf.get_variable(name='W_1',
                      shape=[7, 7, I_1.get_shape()[-1], 64],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
B_1 = tf.get_variable(name='B_1',
                      shape=[64],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
O_1 = tf.nn.relu(tf.nn.conv2d(I_1, W_1, strides=[1, 2, 2, 1], padding='SAME') + B_1)
P_1 = tf.nn.max_pool(O_1, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # Max pool and downsampling

# Setup first chain of resnets
W_2 = tf.get_variable(name='W_2',
                      shape=[1, 1, P_1.get_shape()[-1], blocks[0].num_filters],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
B_2 = tf.get_variable(name='B_2',
                      shape=[blocks[0].num_filters],
                      initializer=tf.truncated_normal_initializer(stddev=0.02))
C_2 = tf.nn.conv2d(P_1, W_2, strides=[1, 1, 1, 1], padding='VALID')
net = C_2 + B_2

# Loop through all res blocks
for block_i, block in enumerate(blocks):
    for repeat_i in range(block.num_repeats):
        name = 'block_%d/repeat_%d' % (block_i, repeat_i)

        with tf.variable_scope(name + '/conv_in'):
            w = tf.get_variable('w', [1, 1, net.get_shape()[-1], block.bottleneck_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('b', [block.bottleneck_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            conv1 = tf.nn.relu(tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='VALID') + b)

        with tf.variable_scope(name + '/conv_bottleneck'):
            w = tf.get_variable('w', [3, 3, conv1.get_shape()[-1], block.bottleneck_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('b', [block.bottleneck_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1], padding='SAME') + b)

        with tf.variable_scope(name + '/conv_out'):
            w = tf.get_variable('w', [1, 1, conv2.get_shape()[-1], block.num_filters],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('b', [block.num_filters],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, w, strides=[1, 1, 1, 1], padding='VALID') + b)

        net2 = conv3 + net
    try:
        # upscale to the next block size
        next_block = blocks[block_i + 1]

        with tf.variable_scope('block_%d/conv_upscale' % block_i):
            w = tf.get_variable('w', [1, 1, net2.get_shape()[-1], next_block.num_filters],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            net = tf.nn.conv2d(net2, w, strides=[1, 1, 1, 1], padding='SAME')
    except IndexError:
        pass

O_11 = tf.nn.avg_pool(net,
                      ksize=[1, net.get_shape().as_list()[1], net.get_shape().as_list()[2], 1],
                      strides=[1, 1, 1, 1], padding='VALID')

O_12 = tf.reshape(O_11, [-1,
                         O_11.get_shape().as_list()[1] *
                         O_11.get_shape().as_list()[2] *
                         O_11.get_shape().as_list()[3]])

shape = O_12.get_shape().as_list()
matrix = tf.get_variable('Matrix', [shape[1], dim_out], tf.float32, tf.random_normal_initializer(stddev=0.02))

O_o = tf.nn.softmax(tf.matmul(O_12, matrix))

'''' ...... drnn end '''
'''||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'''


cross_entropy = -tf.reduce_sum(Y_holder * tf.log(O_o))  # Define loss/eval/training functions
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(O_o, 1), tf.argmax(Y_holder, 1))  # Monitor accuracy
acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

batch_size = 50
epochs = 2
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch_i in range(epochs):

        ''' Training '''
        train_acc = 0
        for _i in range(mnist.train.num_examples // batch_size):
            batch_train_xs, batch_train_ys = mnist.train.next_batch(batch_size)
            train_acc += sess.run([optimizer, acc], feed_dict={X_holder: batch_train_xs,
                                                               Y_holder: batch_train_ys})[1]

        train_acc /= (mnist.train.num_examples // batch_size)

        ''' Validation '''
        valid_acc = 0
        for _j in range(mnist.validation.num_examples // batch_size):
            batch_valid_xs, batch_valid_ys = mnist.validation.next_batch(batch_size)
            valid_acc += sess.run(acc, feed_dict={X_holder: batch_valid_xs,
                                                  Y_holder: batch_valid_ys})

        valid_acc /= (mnist.validation.num_examples // batch_size)
        print('epoch:', epoch_i, ', train acc:', train_acc, ', valid acc:', valid_acc)
print('Using time:', time.time() - start_time)
