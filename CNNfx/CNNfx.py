# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import os

"""Hyperparameters"""
num_filt_1 = 16  # Number of filters in first conv layer
num_filt_2 = 14  # Number of filters in second conv layer
num_fc_1 = 40  # Number of neurons in hully connected layer
max_iterations = 20001
batch_size = 64
keep = 0.4  # Dropout rate in the fully connected layer
learning_rate = 1e-5
input_norm = False  # Do you want z-score input normalization?
Train = True
BackTest = False

"""Load the data"""
data_train = np.loadtxt('TRAIN', delimiter=',')
data_test_val = np.loadtxt('TEST', delimiter=',')
data_test, data_val = np.split(data_test_val, 2)
# Usually, the first column contains the target labels
train_inputs = data_train[:, 1:]
val_inputs = data_val[:, 1:]
test_inputs = data_test[:, 1:]

train_labels = data_train[:, 0]
val_labels = data_val[:, 0]
test_labels = data_test[:, 0]

num_samples_train = train_inputs.shape[0]
dim_in = train_inputs.shape[1]
num_classes = len(np.unique(train_labels))
epochs = np.floor(batch_size * max_iterations / num_samples_train)

print('num_samples_train: %s   dim_in: %s' % (num_samples_train, dim_in))
print('Train: %d epochs' % epochs)

# Organize the classes
base = np.min(train_labels)  # Check if data is 0-based
if base != 0:
    train_labels -= base
    val_labels -= base
    test_labels -= base

if input_norm:
    mean = np.mean(train_inputs, axis=0)
    variance = np.var(train_inputs, axis=0)
    train_inputs -= mean
    # The 1e-9 avoids dividing by zero
    train_inputs /= np.sqrt(variance) + 1e-9
    val_inputs -= mean
    val_inputs /= np.sqrt(variance) + 1e-9
    test_inputs -= mean
    test_inputs /= np.sqrt(variance) + 1e-9

# Nodes for the input variables
input_holder = tf.placeholder("float", shape=[None, dim_in], name='input_holder')
label_holder = tf.placeholder(tf.int64, shape=[None], name='label_holder')
keep_holder = tf.placeholder("float", name='keep_holder')
bn_train_holder = tf.placeholder(tf.bool, name='bn_train_holder')  # Boolean value to guide batchnorm

x_image = tf.reshape(input_holder, [-1, dim_in, 1, 1])
initializer = tf.contrib.layers.xavier_initializer()

''' Build the graph '''
with tf.name_scope("Conv1"):
    W_conv1 = tf.get_variable("Conv_Layer_1", shape=[5, 1, 1, num_filt_1], initializer=initializer)
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[num_filt_1]), name='bias_Conv1')
    a_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

with tf.name_scope('BN_conv1'):
    bn_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=bn_train_holder, updates_collections=None)
    h_bn1 = tf.nn.relu(bn_conv1)

with tf.name_scope("Conv2"):
    W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4, 1, num_filt_1, num_filt_2], initializer=initializer)
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[num_filt_2]), name='bias_Conv2')
    a_conv2 = tf.nn.conv2d(h_bn1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

with tf.name_scope('BN_conv2'):
    bn_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=bn_train_holder, updates_collections=None)
    h_bn2 = tf.nn.relu(bn_conv2)

with tf.name_scope("FC_1"):
    W_fc1 = tf.get_variable("W_fc1", shape=[dim_in * num_filt_2, num_fc_1], initializer=initializer)
    b_fc1 = tf.Variable(tf.constant(0.01, shape=[num_fc_1]), name='b_fc1')
    h_conv3_flat = tf.reshape(h_bn2, [-1, dim_in * num_filt_2])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

with tf.name_scope("FC_2"):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_holder)
    W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_classes], initializer=initializer)
    b_fc2 = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # net output

with tf.name_scope("SoftMax"):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2, label_holder)
    cost = tf.reduce_sum(loss) / batch_size
    loss_summ = tf.scalar_summary("cross entropy_loss", cost)

with tf.name_scope("train"):
    tvars = tf.trainable_variables()
    # We clip the gradients to prevent explosion
    grads = tf.gradients(cost, tvars)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients)
    # The following block plots for every trainable variable
    #  - Histogram of the entries of the Tensor
    #  - Histogram of the gradient over the Tensor
    #  - Histogram of the grradient-norm over the Tensor
    numel = tf.constant([[0]])
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        numel += tf.reduce_sum(tf.size(variable))

        h1 = tf.histogram_summary(variable.name, variable)
        h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)
        h3 = tf.histogram_summary(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

with tf.name_scope("Evaluating_accuracy"):
    correct_prediction = tf.equal(tf.argmax(h_fc2, 1), label_holder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# Define one op to call all summaries
merged = tf.merge_all_summaries()


def load_net(_sess):
    if os.path.exists('log/my_net.ckpt'):
        print('exists net, will load net parameters')
        tf.train.Saver().restore(sess=_sess, save_path='log/my_net.ckpt')
        print('load parameters success')
    else:
        print('no net exists, will train new net')
    return 0


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

softmax = lambda _var: np.exp(_var) / np.sum(np.exp(_var), axis=0)


def train_procession(_sess):
    print('this is train operation !!!')
    writer = tf.train.SummaryWriter('log', _sess.graph)
    _sess.run(tf.initialize_all_variables())

    # load net
    load_net(_sess)

    step = 0  # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):
        batch_ind = np.random.choice(num_samples_train, batch_size, replace=False)

        if i == 0:
            # Use this line to check before-and-after test accuracy
            result = _sess.run(accuracy, feed_dict={input_holder: test_inputs,
                                                    label_holder: test_labels,
                                                    keep_holder: 1.0,
                                                    bn_train_holder: False})
            acc_test_before = result
        if i % 200 == 0:
            # Check training performance
            result = _sess.run([cost, accuracy], feed_dict={input_holder: train_inputs,
                                                            label_holder: train_labels,
                                                            keep_holder: 1.0,
                                                            bn_train_holder: False})
            perf_collect[1, step] = acc_train = result[1]
            cost_train = result[0]

            # Check validation performance
            result = _sess.run([accuracy, cost, merged], feed_dict={input_holder: val_inputs,
                                                                    label_holder: val_labels,
                                                                    keep_holder: 1.0,
                                                                    bn_train_holder: False})
            perf_collect[0, step] = acc_val = result[0]
            cost_val = result[1]

            # Write information to TensorBoard
            writer.add_summary(result[2], i)
            writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file
            print("At %5.0f/%5.0f  Cost...[train: %5.3f  val: %5.3f]  Acc...[train: %5.3f  val: %5.3f]"
                  % (i, max_iterations, cost_train, cost_val, acc_train, acc_val))
            step += 1
        _sess.run(train_step, feed_dict={input_holder: train_inputs[batch_ind],
                                         label_holder: train_labels[batch_ind],
                                         keep_holder: keep,
                                         bn_train_holder: True})
    result = _sess.run([accuracy, numel], feed_dict={input_holder: test_inputs,
                                                     label_holder: test_labels,
                                                     keep_holder: 1.0,
                                                     bn_train_holder: False})
    acc_test = result[0]
    print('The network has %s trainable parameters' % (result[1]))
    print('Train Finished!')

    save_path = tf.train.Saver().save(_sess, 'log/my_net.ckpt')  # save net
    print('Save to:', save_path)

    """Additional plots"""
    print('The accuracy on the test data is %.3f, before training was %.3f' % (acc_test, acc_test_before))
    plt.figure()
    plt.plot(perf_collect[0], label='Valid accuracy')
    plt.plot(perf_collect[1], label='Train accuracy')
    plt.axis([0, step - 1, 0, np.max(perf_collect)])
    plt.legend()
    plt.show()

    # load_tensorboard('log')
    return 0


def back_test_procession(_sess):
    print('this is predict operation !!!')
    _sess.run(tf.initialize_all_variables())
    # load net
    load_net(_sess)

    a1 = [-3.7883, 6.4407, 5.1232, 2.5096, 3.3153, 3.8406, 2.1552, 3.8766, 0.55248, 0.90957, -0.26156, -0.23198,
          -0.22838, -0.22838, 0.21467, -0.028181, 0.041709, -0.24684, -0.30478, -0.24242, -0.22838, -0.22844,
          -0.22836, -0.22838, -0.22838, -0.22838, 3.6434, 1.3842, 1.7943, -0.069181, -0.27715, -0.22838, -0.22841,
          -0.228, -0.22838, -0.22838, -0.22838, -0.22838, -0.32899, -0.070439, -0.21025, -0.33941, -0.24357,
          -0.22839, -0.22841, -0.22838, -0.22839, -0.22838, -0.22838, -0.22838, -0.2284, -0.057723, -0.28235,
          -0.26934, -0.22841, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838,
          -0.21889, -0.28498, -0.22588, -0.22849, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838,
          -0.22851, -0.22846, -0.22842, -0.2284, -0.22838, -0.22784, -0.22838, -0.22835, -0.22838, -0.017036,
          -0.11796, -0.33322, -0.32083, -0.22856, -0.22839, -0.22837, -0.22838, -0.22838, -0.22838, -0.22838,
          -0.22838, -0.22838, -0.2663, -0.27807, -0.22849, -0.22839, -0.22838, -0.22838, -0.22838, -0.31288,
          -0.22618, -0.22585, -0.27105, -0.22856, -0.2285, -0.22838, -0.22838, -0.22838, -0.22838, -0.22816,
          -0.22851, -0.22851, -0.22836, -0.22838, -0.22838, -0.22837, -0.2284, -0.22835, -0.22837, -0.22838,
          -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838, -0.22838,
          -0.22838, -0.22838, -0.22839, -0.22839, -0.2294, 0.065618, 1.8425, -0.22865, -0.22794, -0.27439, -0.22835,
          -0.22838, -0.22791, -0.22838, -0.2283, -0.22727, -0.28902, 0.21675, -0.22642, -0.22892, -0.23819,
          -0.40774, -0.22845, -0.22846, -0.32982, -0.22471, -0.22838, -0.23403, -0.22838, -0.22839, -0.22838,
          -0.22838, -0.22838]
    a1 = np.array([a1])
    pred = _sess.run(h_fc2, feed_dict={input_holder: a1,
                                       keep_holder: 1.0,
                                       bn_train_holder: False})
    pred = pred[0]
    prob = softmax(pred)
    print('output:', pred)
    print('probability:', prob)
    print('type:', np.argmax(pred))

    return 0


# Collect performances in a Numpy array. In future, hope TensorBoard allows more flexibility in plotting
perf_collect = np.zeros((3, int(np.floor(max_iterations / 100))))

sess = tf.Session()  # Main Session start !!!
if Train: train_procession(sess)
if BackTest: back_test_procession(sess)
sess.close()  # Main Session end

'''
data_train = np.loadtxt('TRAIN', delimiter=',')
data_test_val = np.loadtxt('TEST', delimiter=',')

At 20000/20001  Cost...[train: 3.478  val: 20.582]  Acc...[train: 0.831  val: 0.724]

The accuracy on the test data is 0.694, before training was 0.251
'''