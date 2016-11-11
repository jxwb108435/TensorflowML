import tensorflow as tf
import numpy as np

with tf.device('/gpu:0'):
    char_rdic = ['h', 'e', 'l', 'o']  # id -> char
    char_dic = {w: i for i, w in enumerate(char_rdic)}  # char -> id
    x_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype='f')
    sample = [char_dic[c] for c in "hello"]

    # Configuration
    char_vocab_size = len(char_dic)
    rnn_size = char_vocab_size  # 1 hot coding (one of 4)
    time_step_size = 4  # 'hell' -> predict 'ello'
    batch_size = 1  # one sample

    # RNN model
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
    state = tf.zeros([batch_size, rnn_cell.state_size])
    X_split = tf.split(0, time_step_size, x_data)
    outputs, state = tf.nn.rnn(rnn_cell, X_split, state)
    # outputs, state = tf.nn.dynamic_rnn(rnn_cell, X_split, state)

    # logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
    # targets: list of 1D batch-sized int32 Tensors of the same length as logits.
    # weights: list of 1D batch-sized float-Tensors of the same length as logits.
    logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
    targets = tf.reshape(sample[1:], [-1])
    weights = tf.ones([time_step_size * batch_size])

    loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
    cost = tf.reduce_sum(loss) / batch_size
    train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(101):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))

        if 0 == i % 10:
            print(result, [char_rdic[t] for t in result])
