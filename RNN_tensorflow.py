#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNN using tensorflow

Refer to https://www.tensorflow.org/tutorials/sequences/text_generation
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.legacy_seq2seq as seq2seq

_MODEL_NAME = 'model_rnn'
_TRAIN_FILE = tf.keras.utils.get_file('shakespeare.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


def recurrent_neural_network(inputs):
    batch_size, num_steps, feature_dim = inputs.shape
    keep_prob = 0.6
    with tf.variable_scope("recurrent_neural_network", reuse=None):
        # The rnn_cell is similar to the layers in other neural network
        # Here, we use 3 layers to form a RNN. Note the neuron number of output layer should be equal to the class
        # number of the data set if we do not use full connected layer later.
        rnn_cell1 = tf.nn.rnn_cell.GRUCell(num_units=64, activation=tf.nn.leaky_relu)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell1, output_keep_prob=keep_prob)

        rnn_cell2 = tf.nn.rnn_cell.GRUCell(num_units=128, activation=tf.nn.leaky_relu)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell2, output_keep_prob=keep_prob)

        rnn_cell3 = tf.nn.rnn_cell.GRUCell(num_units=64, activation=tf.nn.sigmoid)
        cell3 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell3, output_keep_prob=keep_prob)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])
        # Initial state of the RNN memory.
        state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)

        # output size is [batch_size, num_steps, num_unit], here is [32, 100, 64]
        rnn_outputs, final_s = tf.nn.dynamic_rnn(
            multi_rnn_cell,  # cell you have chosen
            inputs,  # input
            initial_state=state,  # the initial hidden state
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
        )

        # The num_units here must be the same as the num_units of the output layer of the RNN cell
        num_units = 64
        with tf.variable_scope('rnn'):
            # w size: [64, 65]
            w = tf.get_variable("softmax_w", [num_units, output_dimension])
            # b size: [65]
            b = tf.get_variable("softmax_b", [output_dimension])

        # [y00, y01, y02]   [w00, w01, w02, w03, w04]       [y00 * w00 + y01 * w10 + y02 * w20, y00 * w01 + y01 * w11 + y02 * w21, ...
        # [y10, y11, y12]   [w10, w11, w12, w13, w14]       [y10 * w00 + y11 * w10 + y12 * w20, ...
        # [y20, y21, y22]   [w20, w21, w22, w23, w24]  ==>  [y20 * w00 + y21 * w10 + y22 * w20, ...
        # [y30, y31, y32]                                   [y30 * w00 + y31 * w10 + y32 * w20, ...
        # Above is an example of matrix multiply of y and w.
        # For each row of y, we can assume it as the output of each step in RNN. So [y00, y01, y02] is the earliest output
        # For each row of the result (i.e. logits), it is only impacted by corresponding row of y, while all the weights
        # in w are involved in calculation. We can assume it as one full connected layer (in above example, it has 5
        # neurons, i.e. each column of w are the weights of one neuron)
        # So, for output of each step in RNN, we send them to the same full connected layer, and then get the prediction
        # for each step.
        with tf.name_scope('fc'):
            # y size: [3200, 64]
            y = tf.reshape(rnn_outputs, [-1, num_units])
            # logits size: [3200, 65]
            logits = tf.matmul(y, w) + b

        # <Note>: It is totally same to use below commented codes to replace above part!
        # logits = tf.layers.dense(rnn_outputs, units=output_dimension, activation=None)
        # logits = tf.reshape(logits, [-1, output_dimension])

        # Outputting the logits is really important here, because we need to use it to calculate loss
        # fixme: why using prob to calculate loss cannot get good results?
        with tf.name_scope('softmax'):
            prob = tf.nn.softmax(logits)

        return logits, prob


if __name__ == '__main__':
    # Read training data
    text = open(_TRAIN_FILE).read()
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    # Hyperparameters
    batch_size = 32
    num_steps = 100
    learning_rate = 0.0005
    steps = 30000
    output_dimension = len(vocab)

    # Placeholder for the inputs in a given iteration.
    words = tf.placeholder(tf.float32, [batch_size, None, 1])
    labels = tf.placeholder(tf.int32, [batch_size, None])
    outputs, prob = recurrent_neural_network(words)

    targets = tf.reshape(labels, [-1])
    with tf.name_scope('loss'):
        targets = tf.reshape(labels, [-1])
        _loss = seq2seq.sequence_loss_by_example([outputs],
                                                 [targets],
                                                 [tf.ones_like(targets, dtype=tf.float32)])
        loss = tf.reduce_mean(_loss)

    # <Note>: It is same to use below commented line to get the loss
    # loss = tf.losses.sparse_softmax_cross_entropy(targets, outputs)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('loss', loss)  # For tensorboard
    # Merge all the summaries, but here is just one loss scalar
    merged = tf.summary.merge_all()  # For tensorboard
    # Save the graph and model used by tensorboard
    file_writer = tf.summary.FileWriter(_MODEL_NAME, sess.graph)  # For tensorboard


    # Try to generate some text using the trained model
    def generate_text(length):
        result = ''
        # Specify the initial inputs
        idx = np.random.randint(0, len(text_as_int) - num_steps - 1)
        test_words = text_as_int[idx: idx + num_steps]
        assert length > 0

        # As the test words are filled with predicted chars, we just move the inputs forward with one more char
        for _ in range(length):
            predict_vector = sess.run(outputs, feed_dict={
                words: np.reshape(test_words, [1, num_steps, 1]),
            })
            predict_int = np.argmax(predict_vector[0][-1])
            result += idx2char[predict_int]
            test_words[:-1] = test_words[1:]
            test_words[-1] = predict_int

        return result


    for _s in range(steps):
        start_indexes = np.random.randint(0, len(text_as_int) - num_steps - 1, size=batch_size)
        input_example = []
        output_labels = []
        for idx in start_indexes:
            input_example.append(text_as_int[idx: idx + num_steps])
            output_labels.append(text_as_int[idx + 1: idx + num_steps + 1])

        loss_value, summary = sess.run([loss, merged], feed_dict={
            words: np.reshape(input_example, [batch_size, num_steps, 1]),
            labels: output_labels,
        })
        file_writer.add_summary(summary, _s)  # For tensorboard
        # Train the weights
        sess.run(optimizer, feed_dict={
            words: np.reshape(input_example, [batch_size, num_steps, 1]),
            labels: output_labels,
        })

        if _s % 100 == 0:
            print('====================step {}====================\n'.format(_s))
            predict_vector = sess.run(prob, feed_dict={
                words: np.reshape(input_example, [batch_size, num_steps, 1]),
                labels: output_labels,
            })
            result = ''
            for v in predict_vector[:num_steps, :]:
                result += idx2char[np.argmax(v)]
            print(result)
            # print(generate_text(100))
