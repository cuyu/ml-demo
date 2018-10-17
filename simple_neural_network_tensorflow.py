#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Construct a simple neural network using tensorflow

Note:
    1. To debug, pls uncomment line 58 and run tensorboard first, then run the python file:
        tensorboard \
        --logdir ~/Code/Python/ml-demo/model_simple_nn \
        --port 6006 \
        --debugger_port 6064
    2.To see the training process,
"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np


def learn_tensorflow_basics():
    # a, b, c, d here are called 'tensor', which are similar to `Unit` or `Gate` in our program
    # With tensors, we can define the neural network structure
    a = tf.constant([5, 3], name='input_a')
    b = tf.reduce_prod(a, name='prod_b')
    c = tf.reduce_sum(a, name='sum_c')
    d = tf.add(b, c, name='add_d')

    # tensors will not calculate until we call Session.run
    sess = tf.Session()
    sess.run(a)
    sess.run(b)
    sess.run(c)
    sess.run(d)
    sess.close()


def simple_neural_network(inputs):
    with tf.variable_scope("simple_neural_network", reuse=None):
        # `tf.layers.dense` is just full connected layer
        # Hidden layer 1
        x = tf.layers.dense(inputs, units=2, activation=tf.nn.relu, name='hidden1')
        # Hidden layer 2
        x = tf.layers.dense(x, units=2, activation=tf.nn.relu, name='hidden2')
        # Hidden layer 3, use liner activation function
        x = tf.layers.dense(x, units=1, activation=None, name='hidden3')
        return x


class NeuralNetworkClassifier(object):
    def __init__(self):
        # We hard code the input dimension to 2 and output to 1.
        # `None` in placeholder means the input batch number is uncertain
        self.inputs = tf.placeholder(tf.float32, [None, 2])
        self.labels = tf.placeholder(tf.float32, [None, 1])
        self.outputs = simple_neural_network(self.inputs)
        self.sess = tf.Session()
        # Uncomment below line for debugging
        # self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, 'localhost:6064')

    def train(self, data_set, learning_rate=0.01, steps=100):
        loss = tf.losses.hinge_loss(self.labels, self.outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        input_data = []
        input_label = []
        for data, label in data_set:
            input_data.append(data)
            input_label.append(label)

        input_data = np.array(input_data, dtype=np.float32)
        input_label = np.array(input_label, dtype=np.float32)

        saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        for i in range(steps):
            loss_value = self.sess.run(loss, feed_dict={
                self.inputs: input_data,
                self.labels: input_label,
            })
            print(loss_value)
            self.sess.run(optimizer, feed_dict={
                self.inputs: input_data,
                self.labels: input_label,
            })

    def predict(self, x, y):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: [[x, y]],
        })


if __name__ == '__main__':
    data_set = [
        ([1.2, 0.7], [1]),
        ([-0.3, -0.5], [-1]),
        ([3.0, 0.1], [1]),
        ([-0.1, -1.0], [-1]),
        ([-1.0, 1.1], [-1]),
        ([2.1, -3.0], [1]),
    ]
    classifier = NeuralNetworkClassifier()
    classifier.train(data_set)
    classifier.predict(0, 0)
