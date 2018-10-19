#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN using tensorflow

Can also refer to https://www.tensorflow.org/tutorials/estimators/cnn
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

MNIST = input_data.read_data_sets('MNIST_data')
_MODEL_NAME = 'model_cnn'


def convolutional_neural_network(inputs):
    with tf.variable_scope("CNN", reuse=None):
        x = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        # CONV layer 1
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv1')
        x = tf.layers.dropout(x, rate=0.4)
        # POOLING layer 1, the input size from 28x28 -> 14x14
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='pooling1')
        # CONV layer 2
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2')
        x = tf.layers.dropout(x, rate=0.4)
        # CONV layer 3
        x = tf.layers.conv2d(x, filters=8, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3')
        # POOLING layer 2, the input size from 14x14 -> 7x7
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='pooling2')
        # Flatten the input to 7x7x8=392 dimensions
        x = tf.reshape(x, shape=[-1, 7 * 7 * 8])
        # Hidden layer 1
        x = tf.layers.dense(x, units=32, activation=tf.nn.relu, name='hidden1')
        x = tf.layers.dropout(x, rate=0.4)
        # Hidden layer 2
        x = tf.layers.dense(x, units=16, activation=tf.nn.relu, name='hidden2')
        # Hidden layer 3, output 10 classes
        x = tf.layers.dense(x, units=10, activation=tf.nn.sigmoid, name='hidden3')
        return x


class ConvolutionalNeuralNetworkClassifier(object):
    def __init__(self):
        # `None` in placeholder means the input batch number is uncertain
        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.outputs = convolutional_neural_network(self.inputs)
        self.sess = tf.Session()
        # Uncomment below line for debugging
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

    def train(self, data_set, learning_rate=0.001, steps=100):
        with tf.name_scope('loss'):  # For tensorboard
            loss = tf.losses.mean_squared_error(self.labels, self.outputs)
        tf.summary.scalar('loss', loss)  # For tensorboard
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Merge all the summaries, but here is just one loss scalar
        merged = tf.summary.merge_all()  # For tensorboard
        # Save the graph and model used by tensorboard
        file_writer = tf.summary.FileWriter(_MODEL_NAME, self.sess.graph)  # For tensorboard

        batch_size = 64

        self.sess.run(tf.global_variables_initializer())
        for i in range(steps):
            batch, labels = data_set.train.next_batch(batch_size=batch_size)
            input_data = [np.reshape(b, [28, 28, 1]) for b in batch]
            input_label = []
            for l in labels:
                _label = np.zeros([10], dtype=np.float32)
                _label[l] = 1
                input_label.append(_label)
            loss_value, summary, _ = self.sess.run([loss, merged, optimizer], feed_dict={
                self.inputs: input_data,
                self.labels: input_label,
            })
            file_writer.add_summary(summary, i)  # For tensorboard

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: [np.reshape(inputs, [28, 28, 1])],
        })


if __name__ == '__main__':
    classifier = ConvolutionalNeuralNetworkClassifier()
    classifier.train(MNIST, steps=50000)
    batch, labels = MNIST.test.next_batch(batch_size=10000)
    correct = 0
    for i in range(len(batch)):
        predict_array = classifier.predict(batch[i])
        predict_number = np.argmax(predict_array)
        if predict_number == labels[i]:
            correct += 1
    print('Accuracy: {}'.format(correct/len(batch)))
