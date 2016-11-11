import tensorflow as tf
import os
import pickle
import numpy as np
import pandas as pd
import datetime


def tf_confusion_metrics(_output, _labels_holder, _sess, _feed_dict):
    predictions = tf.argmax(_output, 1)
    actuals = tf.argmax(_labels_holder, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = _sess.run([tp_op, tn_op, fp_op, fn_op], _feed_dict)

    tpr = float(tp) / (float(tp) + float(fn))

    _accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', _accuracy)


def relu_weights(_n1, _n2):
    _var = (2.0 / (_n1 * _n2)) ** 0.5
    return tf.Variable(tf.truncated_normal([_n1, _n2], stddev=0.0001) * _var)


dim_in = 15
dim_out = 2

inputs_holder = tf.placeholder(tf.float32, [None, dim_in])
labels_holder = tf.placeholder(tf.float32, [None, dim_out])

# layer 1:    hidden layer 1
N1 = 40
W1 = relu_weights(dim_in, N1)
B1 = tf.Variable(tf.constant(0.01, shape=[N1]))
I1 = inputs_holder
O1 = tf.nn.relu(tf.matmul(I1, W1) + B1)

# layer 2:    hidden layer 2
N2 = 40
W2 = relu_weights(N1, N2)
B2 = tf.Variable(tf.constant(0.01, shape=[N2]))
O2 = tf.nn.relu(tf.matmul(O1, W2) + B2)

# layer 3:    hidden layer 3
N3 = 40
W3 = relu_weights(N2, N3)
B3 = tf.Variable(tf.constant(0.01, shape=[N3]))
O3 = tf.nn.relu(tf.matmul(O2, W3) + B3)

# layer 4:    output_layer
N4 = dim_out
W4 = tf.Variable(tf.truncated_normal([N3, N4], stddev=0.0001))
B4 = tf.Variable(tf.constant(0.01, shape=[N4]))
I4 = O3
O4 = tf.matmul(O3, W4) + B4

output_softmax = tf.nn.softmax(O4)

cost = -tf.reduce_sum(labels_holder * tf.log(output_softmax))
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(output_softmax, 1), tf.argmax(labels_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# load data using pickle
pkl_file = open("data_co.pkl", 'rb')
pd_data = pickle.load(pkl_file)
pkl_file.close()
print('load data successful !!!')

all_inputs_df = pd_data[pd_data.columns[5:]]  # [1, 2, 3, 4, 5][2:] -> [3, 4, 5]  except 'date'
all_labels_df = pd_data[pd_data.columns[3:5]]  # [1, 2, 3, 4, 5][:2] -> [1, 2]  except 'date'

train_inputs_df = all_inputs_df['2010-01-01':'2016-10-13']  # between and include '2016-01-01' '2016-01-01'
train_labels_df = all_labels_df['2010-01-01':'2016-10-13']

test_inputs_df = all_inputs_df['2016-01-02':'2016-10-12']  # after and include '2016-01-02'
test_labels_df = all_labels_df['2016-01-02':'2016-10-12']

train_dict = {inputs_holder: train_inputs_df.values,
              labels_holder: train_labels_df.values.reshape(len(train_labels_df.values), dim_out)}

test_dict = {inputs_holder: test_inputs_df.values,
             labels_holder: test_labels_df.values.reshape(len(test_labels_df.values), dim_out)}

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(1, 5001):
        sess.run(train_op, feed_dict=train_dict)
        if 0 == i % 500:
            print(i, sess.run(accuracy, feed_dict=train_dict))
    # tf_confusion_metrics(output_softmax, labels_holder, sess, test_dict)

    print(sess.run(output_softmax, feed_dict={inputs_holder: [all_inputs_df.ix['2016-10-14'].values]}))

