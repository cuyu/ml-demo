#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNN using tensorflow

Refer to https://www.tensorflow.org/tutorials/sequences/text_generation
"""
import tensorflow as tf
import numpy as np

_MODEL_NAME = 'model_rnn'
_TRAIN_FILE = tf.keras.utils.get_file('shakespeare.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


def recurrent_neural_network(inputs):
    # cell_size = 32
    batch_size, num_steps, feature_dim = inputs.shape
    with tf.variable_scope("recurrent_neural_network", reuse=None):
        # The rnn_cell is similar to the layers in other neural network
        # Here, we use 3 layers to form a RNN. Note the neuron number of output layer should be equal to the class
        # number of the data set.
        rnn_cell1 = tf.nn.rnn_cell.GRUCell(num_units=64, activation=tf.nn.leaky_relu)
        rnn_cell2 = tf.nn.rnn_cell.GRUCell(num_units=128, activation=tf.nn.leaky_relu)
        rnn_cell3 = tf.nn.rnn_cell.GRUCell(num_units=output_dimension, activation=tf.nn.sigmoid)
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell1, rnn_cell2, rnn_cell3])
        # Initial state of the LSTM memory.
        state = (rnn_cell1.zero_state(batch_size, dtype=tf.float32),
                 rnn_cell2.zero_state(batch_size, dtype=tf.float32),
                 rnn_cell3.zero_state(batch_size, dtype=tf.float32),)
        # for i in range(num_steps):
        #     output, state = rnn_cell(inputs[:, i, :], state)
        # final_state = state

        # Add a full connected layer for prediction
        # x = tf.layers.dense(final_state, units=output_dimension, activation=tf.nn.sigmoid)

        outputs, final_s = tf.nn.dynamic_rnn(
            multi_rnn_cell,  # cell you have chosen
            inputs,  # input
            initial_state=state,  # the initial hidden state
            time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
        )

        return outputs


if __name__ == '__main__':
    # Read training data
    text = open(_TRAIN_FILE).read()
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    # Hyperparameters
    batch_size = 1
    num_steps = 100
    learning_rate = 0.0001
    steps = 20000
    output_dimension = len(vocab)

    # Placeholder for the inputs in a given iteration.
    words = tf.placeholder(tf.float32, [batch_size, None, 1])
    labels = tf.placeholder(tf.float32, [batch_size, None, output_dimension])
    rnn = recurrent_neural_network(words)
    loss = tf.losses.sigmoid_cross_entropy(labels, rnn)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('loss', loss)  # For tensorboard
    # Merge all the summaries, but here is just one loss scalar
    merged = tf.summary.merge_all()  # For tensorboard
    # Save the graph and model used by tensorboard
    file_writer = tf.summary.FileWriter(_MODEL_NAME, sess.graph)  # For tensorboard

    for _s in range(steps):
        start_index = np.random.randint(0, len(text_as_int) - num_steps - 1)
        input_example = text_as_int[start_index: start_index + num_steps]
        output_labels = text_as_int[start_index + 1: start_index + num_steps + 1]
        label_vector = np.zeros([batch_size, num_steps, output_dimension])
        for i in range(len(output_labels)):
            label_vector[0][i][output_labels[i]] = 1
        loss_value, summary = sess.run([loss, merged], feed_dict={
            words: np.reshape(input_example, [batch_size, num_steps, 1]),
            labels: label_vector,
        })
        file_writer.add_summary(summary, _s)  # For tensorboard
        # Train the weights
        sess.run(optimizer, feed_dict={
            words: np.reshape(input_example, [batch_size, num_steps, 1]),
            labels: label_vector,
        })

    # Try to generate some text using the trained model
    def generate_text(first_char, length):
        result = ''
        # Specify the initial inputs as 0 (maybe we should set he first char as a random number~)
        test_words = np.zeros(num_steps)
        test_words[0] = char2idx[first_char]
        assert length >= 100
        for i in range(num_steps - 1):
            predict_vector = sess.run(rnn, feed_dict={
                words: np.reshape(test_words, [batch_size, num_steps, 1]),
            })
            # For the inputs here, we only pick the predict char next
            predict_int = np.argmax(predict_vector[0][i])
            # Reset the test words using the predicted char
            test_words[i + 1] = predict_int
            result += idx2char[predict_int]

        # As the test words are filled with predicted chars, we just move the inputs forward with one more char
        for _ in range(length - num_steps):
            test_words[:-1] = test_words[1:]
            test_words[-1] = predict_int
            predict_vector = sess.run(rnn, feed_dict={
                words: np.reshape(test_words, [batch_size, num_steps, 1]),
            })
            predict_int = np.argmax(predict_vector[0][-1])
            result += idx2char[predict_int]

        return result


    print(generate_text('Q', 200))
