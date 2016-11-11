# coding=utf-8
# -*- coding:cp936 -*-
import tensorflow as tf
import numpy as np
import os
import pandas as pd

"""Load the data"""
datingSet = np.loadtxt('dating.txt')
df = pd.DataFrame(datingSet, columns=['fly', 'game', 'icecream', 'labels'])

# normalize data
df['fly'] = (df['fly'] - df['fly'].min()) / (df['fly'].max() - df['fly'].min())
df['game'] = (df['game'] - df['game'].min()) / (df['game'].max() - df['game'].min())
df['icecream'] = (df['icecream'] - df['icecream'].min()) / (df['icecream'].max() - df['icecream'].min())

df_random = df.take(np.random.permutation(len(df)))  # random resort origin df for choice train, val, test data

df_train = df_random[0:800]
df_val = df_random[800:900]
df_test = df_random[900:1000]

train_inputs = df_train[['fly', 'game', 'icecream']].values
train_labels = df_train['labels'].values

val_inputs = df_val[['fly', 'game', 'icecream']].values
val_labels = df_val['labels'].values

test_inputs = df_test[['fly', 'game', 'icecream']].values
test_labels = df_test['labels'].values

# Organize the classes
base = np.min(train_labels)  # Check if data is 0-based
if base != 0:
    train_labels -= base
    val_labels -= base
    test_labels -= base

''' using tensorflow to translated label as one_hot '''
sess = tf.Session()
train_labels = sess.run(tf.one_hot(indices=train_labels.astype(int), depth=3))
val_labels = sess.run(tf.one_hot(indices=val_labels.astype(int), depth=3))
test_labels = sess.run(tf.one_hot(indices=test_labels.astype(int), depth=3))
print('translated label as one_hot')
sess.close()

# Network Parameters
n_hidden_1 = 12  # layer1 num features
n_hidden_2 = 12  # layer2 num features
n_hidden_3 = 12  # layer3 num features
n_hidden_4 = 12  # layer4 num features
n_input = 3  # MNIST data input (img shape: 28*28)
n_classes = 3  # MNIST total classes (0-9 digits)
batch_size = 1

train_flags = True

input_holder = tf.placeholder(tf.float32, [None, n_input])
label_holder = tf.placeholder(tf.float32, [None, n_classes])
keep_holder = tf.placeholder(tf.float32)


def load_net(_saver):
    if os.path.exists('mnistmlp/net.ckpt'):
        print('exists net, will load net parameters')
        _saver.restore(sess, 'mnistmlp/net.ckpt')
        print('load parameters success')
    else:
        print('no net exists, will train new net')
    return 0


def relu_weights(_n1, _n2):
    _var = (2.0 / (_n1 * _n2)) ** 0.5
    return tf.Variable(tf.random_normal([_n1, _n2]) * _var)


def fc_with_dropout_relu(_input, _weights, _biases, _keep_holder):
    _layer_input = tf.nn.dropout(_input, keep_prob=_keep_holder)
    _layer_output = tf.nn.relu(tf.add(tf.matmul(_layer_input, _weights), _biases))
    return _layer_output


def multilayer_perceptron(_x, _weights, _biases):
    hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(_x, _weights['h1']), _biases['b1']))

    hidden_layer_2 = fc_with_dropout_relu(hidden_layer_1, _weights['h2'], _biases['b2'], keep_holder)
    hidden_layer_3 = fc_with_dropout_relu(hidden_layer_2, _weights['h3'], _biases['b3'], keep_holder)
    hidden_layer_4 = fc_with_dropout_relu(hidden_layer_3, _weights['h4'], _biases['b4'], keep_holder)

    _net_output = tf.matmul(hidden_layer_4, _weights['out']) + _biases['out']

    return _net_output


weights = {'h1': relu_weights(n_input, n_hidden_1),
           'h2': relu_weights(n_hidden_1, n_hidden_2),
           'h3': relu_weights(n_hidden_2, n_hidden_3),
           'h4': relu_weights(n_hidden_3, n_hidden_4),
           'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))}

biases = {'b1': tf.Variable(tf.constant(0.01, shape=[n_hidden_1])),
          'b2': tf.Variable(tf.constant(0.01, shape=[n_hidden_2])),
          'b3': tf.Variable(tf.constant(0.01, shape=[n_hidden_3])),
          'b4': tf.Variable(tf.constant(0.01, shape=[n_hidden_4])),
          'out': tf.Variable(tf.constant(0.01, shape=[n_classes]))}

pred = multilayer_perceptron(input_holder, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, label_holder))  # Softmax loss
train_set = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)  # Adam Optimizer


def train_processing(_sess):
    for epoch in range(10):
        avg_cost = 0.
        total_batch = int(train_inputs.shape[0] / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_ind = np.random.choice(train_inputs.shape[0], batch_size, replace=False)
            # Fit training using batch data
            _sess.run(train_set, feed_dict={input_holder: train_inputs[batch_ind],
                                            label_holder: train_labels[batch_ind],
                                            keep_holder: 0.5})
            # Compute average loss
            avg_cost += _sess.run(cost, feed_dict={input_holder: train_inputs[batch_ind],
                                                   label_holder: train_labels[batch_ind],
                                                   keep_holder: 1.0}) / total_batch
        # Display logs per epoch step
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

        # Calculate validation accuracy
        correct_validation = tf.equal(tf.argmax(pred, 1), tf.argmax(label_holder, 1))
        accuracy_validation = tf.reduce_mean(tf.cast(correct_validation, tf.float32))
        acc_validation = _sess.run(accuracy_validation, feed_dict={input_holder: val_inputs,
                                                                   label_holder: val_labels,
                                                                   keep_holder: 1.0})
        print('Accuracy_validation:', acc_validation, '\n' * 2)

    print('Train Finished !', '\n')

    # save net
    save_path = saver.save(_sess, 'mnistmlp/net.ckpt')
    print('Net saved to:', save_path)

    # Calculate test accuracy
    correct_test = tf.equal(tf.argmax(pred, 1), tf.argmax(label_holder, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_test, tf.float32))
    acc_test = _sess.run(accuracy_test, feed_dict={input_holder: test_inputs,
                                                   label_holder: test_labels,
                                                   keep_holder: 1.0})
    print('Accuracy_test:', acc_test)

    return 0


saver = tf.train.Saver()

''' Main Session start !!! '''
sess = tf.Session()
sess.run(tf.initialize_all_variables())
load_net(saver)

if train_flags:
    train_processing(sess)

sess.close()
''' Main Session end !!! '''
