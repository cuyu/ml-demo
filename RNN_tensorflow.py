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
    # cell_size = 32
    batch_size, num_steps, feature_dim = inputs.shape
    keep_prob = 0.6
    with tf.variable_scope("recurrent_neural_network", reuse=None):
        # The rnn_cell is similar to the layers in other neural network
        # Here, we use 3 layers to form a RNN. Note the neuron number of output layer should be equal to the class
        # number of the data set.
        rnn_cell1 = tf.nn.rnn_cell.GRUCell(num_units=64, activation=tf.nn.leaky_relu)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell1, output_keep_prob=keep_prob)
        # rnn_dropout1 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell1, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        rnn_cell2 = tf.nn.rnn_cell.GRUCell(num_units=128, activation=tf.nn.leaky_relu)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell2, output_keep_prob=keep_prob)
        # rnn_dropout2 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell2, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        rnn_cell3 = tf.nn.rnn_cell.GRUCell(num_units=128, activation=tf.nn.sigmoid)
        cell3 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell3, output_keep_prob=keep_prob)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])
        # Initial state of the LSTM memory.
        state = multi_rnn_cell.zero_state(batch_size, dtype=tf.float32)
        # for i in range(num_steps):
        #     output, state = rnn_cell(inputs[:, i, :], state)
        # final_state = state

        outputs, final_s = tf.nn.dynamic_rnn(
            multi_rnn_cell,  # cell you have chosen
            inputs,  # input
            initial_state=state,  # the initial hidden state
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
        )

        # # Add a full connected layer for prediction
        # x = tf.layers.dense(outputs, units=output_dimension, activation=tf.nn.softmax)

        num_units = 128
        with tf.variable_scope('rnn'):
            w = tf.get_variable("softmax_w", [num_units, output_dimension])
            b = tf.get_variable("softmax_b", [output_dimension])

            embedding = tf.get_variable("embedding", [output_dimension, num_units])
            # inputs = tf.nn.embedding_lookup(embedding, inputs)

        with tf.name_scope('fc'):
            y = tf.reshape(outputs, [-1, num_units])
            logits = tf.matmul(y, w) + b

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
    # loss = tf.losses.sparse_softmax_cross_entropy(labels, outputs)
    with tf.name_scope('loss'):
        targets = tf.reshape(labels, [-1])
        _loss = seq2seq.sequence_loss_by_example([outputs],
                                                [targets],
                                                [tf.ones_like(targets, dtype=tf.float32)])
        loss = tf.reduce_mean(_loss)

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
        # label_vector = np.zeros([batch_size, num_steps, output_dimension])
        # for j in range(batch_size):
        #     for i in range(num_steps):
        #         label_vector[j][i][output_labels[j][i]] = 1
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
