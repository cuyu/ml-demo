#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN using tensorflow
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets('MNIST_data')
_MODEL_NAME = 'model_gan'


# Code by Parag Mital (github.com/pkmital/CADL)
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# The generative network can be any network depends on generating target (CNN/DNN/...)
def generative_network(inputs):
    with tf.variable_scope("generative_network", reuse=None):
        x = tf.layers.dense(inputs, units=16, activation=tf.nn.relu, name='g_hidden1')
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.dense(x, units=7 * 7 * 8, activation=tf.nn.relu, name='g_hidden2')
        x = tf.reshape(x, shape=[-1, 7, 7, 8])
        # We use strides=2 so that input size from 7x7 -> 14x14
        x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same',
                                       name='transpose_conv1')
        # The input size from 14x14 -> 28x28
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same',
                                       name='transpose_conv2')
        x = tf.layers.dropout(x, rate=0.4)
        # We use strides=1, so the input size stay the same
        # Note we do not use pooling layer, because pooling will lose some information which makes it difficult to train
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=1, activation=tf.nn.relu, padding='same',
                                       name='transpose_conv3')
        # fixme: is when strides=1 padding='same', conv2d_trainspose == conv2d?
        # Use filters=1 to output 1 dimension image (gray-scale image)
        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same',
                                       name='transpose_conv4')
        return x


# The discriminative network type is determined by generative network, here we use CNN
def discriminative_network(inputs, reuse=None):
    with tf.variable_scope("discriminative_network", reuse=reuse):
        x = tf.layers.conv2d(inputs, filters=64, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv1')
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
        x = tf.layers.dense(x, units=64, activation=tf.nn.relu, name='d_hidden1')
        x = tf.layers.dropout(x, rate=0.4)
        # Hidden layer 2, output 1 unit for 2 classes: fake or real image
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid, name='d_hidden2')
        return x


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    steps = 10000
    learning_rate = 0.0001

    noise = tf.placeholder(tf.float32, [None, 1])
    real_images = tf.placeholder(tf.float32, [None, 28, 28, 1])
    g_network = generative_network(noise)
    d_network_real = discriminative_network(real_images)
    # Reuse the same variables in the discriminative_network above
    d_network_fake = discriminative_network(g_network, reuse=True)
    # The generative_network's purpose is to make the discriminative_network think the fake inputs are real
    loss_g = tf.losses.mean_squared_error(tf.ones_like(d_network_fake), d_network_fake)
    # The discriminative_network's purpose is to make real images get output ones and fake images get output zeros
    loss_d = (tf.losses.mean_squared_error(tf.ones_like(d_network_real), d_network_real) + \
              tf.losses.mean_squared_error(tf.zeros_like(d_network_fake), d_network_fake)) / 2

    optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_g)
    optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_d)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
        # Use random values as input of generative_network
        random_noise = np.random.uniform(0.0, 1.0, [batch_size, 1]).astype(np.float32)
        # The real images to train discriminative_network
        batch, _ = MNIST.train.next_batch(batch_size=batch_size)
        batch_images = [np.reshape(b, [28, 28, 1]) for b in batch]

        loss_g_value, loss_d_value = sess.run([loss_g, loss_d], feed_dict={
            noise: random_noise,
            real_images: batch_images,
        })

        if loss_g_value < loss_d_value:
            # Train discriminative_network
            sess.run(optimizer_d, feed_dict={
                noise: random_noise,
                real_images: batch_images,
            })
        else:
            # Train generative_network
            print('train g')
            sess.run(optimizer_g, feed_dict={
                noise: random_noise,
                real_images: batch_images,
            })

        if i % 50 == 0:
            # Display some images created by generative_network
            gen_img = sess.run(g_network, feed_dict={
                noise: random_noise,
            })
            imgs = [img[:, :, 0] for img in gen_img]
            m = montage(imgs)
            gen_img = m
            plt.axis('off')
            plt.imshow(gen_img, cmap='gray')
            plt.show()
