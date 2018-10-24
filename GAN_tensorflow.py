#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN using tensorflow

Refer to https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/DCGAN-MNIST.ipynb
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


# Use the loss function in original GAN paper
def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


# The generative network can be any network depends on generating target (CNN/DNN/...)
def generative_network(inputs, is_training):
    momentum = 0.99
    with tf.variable_scope("generative_network", reuse=None):
        x = tf.layers.dense(inputs, units=16, activation=tf.nn.leaky_relu, name='g_hidden1')
        x = tf.layers.dense(x, units=7 * 7 * 8, activation=tf.nn.leaky_relu, name='g_hidden2')
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.batch_normalization(x, training=is_training, momentum=momentum)
        x = tf.reshape(x, shape=[-1, 7, 7, 8])
        # We use strides=2 so that input size from 7x7 -> 14x14
        x = tf.layers.conv2d_transpose(x, filters=8, kernel_size=3, strides=2, activation=tf.nn.leaky_relu,
                                       padding='same', name='transpose_conv1')
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.batch_normalization(x, training=is_training, momentum=momentum)
        # The input size from 14x14 -> 28x28
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu,
                                       padding='same', name='transpose_conv2')
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.batch_normalization(x, training=is_training, momentum=momentum)
        # We use strides=1, so the input size stay the same
        # Note we do not use pooling layer, because pooling will lose some information which makes it difficult to train
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=1, activation=tf.nn.leaky_relu,
                                       padding='same', name='transpose_conv3')
        # fixme: is when strides=1 padding='same', conv2d_trainspose == conv2d?
        x = tf.layers.dropout(x, rate=0.4)
        x = tf.layers.batch_normalization(x, training=is_training, momentum=momentum)
        # Use filters=1 to output 1 dimension image (gray-scale image); use sigmoid to make the output between 0 to 1
        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=3, strides=1, activation=tf.nn.sigmoid,
                                       padding='same', name='transpose_conv4')
        return x


# The discriminative network type is determined by generative network, here we use CNN
def discriminative_network(inputs, reuse=None):
    with tf.variable_scope("discriminative_network", reuse=reuse):
        x = tf.layers.conv2d(inputs, filters=64, kernel_size=5, strides=(2, 2), activation=tf.nn.leaky_relu,
                             padding='same', name='conv1')
        x = tf.layers.dropout(x, rate=0.4)
        # CONV layer 2
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', name='conv2')
        x = tf.layers.dropout(x, rate=0.4)
        # CONV layer 3
        x = tf.layers.conv2d(x, filters=8, kernel_size=3, activation=tf.nn.leaky_relu, strides=(2, 2), padding='same',
                             name='conv3')
        # Flatten the input to 7x7x8=392 dimensions
        x = tf.reshape(x, shape=[-1, 7 * 7 * 8])
        # Hidden layer 1
        x = tf.layers.dense(x, units=64, activation=tf.nn.leaky_relu, name='d_hidden1')
        x = tf.layers.dropout(x, rate=0.4)
        # Hidden layer 2, output 1 unit for 2 classes: fake or real image
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid, name='d_hidden2')
        return x


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    noise_dimension = 64
    steps = 10000
    learning_rate = 0.00015

    noise = tf.placeholder(tf.float32, [None, noise_dimension])
    real_images = tf.placeholder(tf.float32, [None, 28, 28, 1])
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    g_network = generative_network(noise, is_training)
    d_network_real = discriminative_network(real_images)
    # Reuse the same variables in the discriminative_network above
    d_network_fake = discriminative_network(g_network, reuse=True)
    # The generative_network's purpose is to make the discriminative_network think the fake inputs are real
    loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_network_fake), d_network_fake))
    # The discriminative_network's purpose is to make real images get output ones and fake images get output zeros
    loss_d = tf.reduce_mean((binary_cross_entropy(tf.ones_like(d_network_real), d_network_real) + \
                             binary_cross_entropy(tf.zeros_like(d_network_fake), d_network_fake)) / 2)

    # It is important to only update corresponding variables when training the networks! (specify var_list in optimizer)
    vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generative_network")]
    vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminative_network")]
    d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
    g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

    # What you specify in the argument to control_dependencies is ensured to be evaluated before anything you define in
    #  the with block (Refer to https://stackoverflow.com/a/42095969/5996843)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Actual loss is the network loss plus weights regularization
        optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_g + g_reg, var_list=vars_g)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_d + d_reg, var_list=vars_d)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('loss_d', loss_d)  # For tensorboard
    tf.summary.scalar('loss_g', loss_g)  # For tensorboard
    # Merge all the summaries, but here is just one loss scalar
    merged = tf.summary.merge_all()  # For tensorboard
    # Save the graph and model used by tensorboard
    file_writer = tf.summary.FileWriter(_MODEL_NAME, sess.graph)  # For tensorboard
    for i in range(steps):
        # Use random values as input of generative_network
        random_noise = np.random.normal(0.0, 1.0, [batch_size, noise_dimension]).astype(np.float32)
        # The real images to train discriminative_network
        batch, _ = MNIST.train.next_batch(batch_size=batch_size)
        batch_images = [np.reshape(b, [28, 28, 1]) for b in batch]

        loss_g_value, loss_d_value, summary = sess.run([loss_g, loss_d, merged], feed_dict={
            noise: random_noise,
            real_images: batch_images,
            is_training: False,
        })
        file_writer.add_summary(summary, i)  # For tensorboard

        # Train discriminative_network
        sess.run(optimizer_d, feed_dict={
            noise: random_noise,
            real_images: batch_images,
            is_training: True,
        })
        # Train generative_network
        sess.run(optimizer_g, feed_dict={
            noise: random_noise,
            real_images: batch_images,
            is_training: True,
        })

        if i % 100 == 0:
            print(loss_d_value, loss_g_value)
            # Display some images created by generative_network
            gen_img = sess.run(g_network, feed_dict={
                noise: random_noise,
                is_training: False,
            })
            imgs = [img[:, :, 0] for img in gen_img]
            m = montage(imgs)
            gen_img = m
            plt.axis('off')
            plt.imshow(gen_img, cmap='gray')
            plt.show()
