#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note:
    1.
"""
import math

import matplotlib.pyplot as plt

from neural_network import Gate, Network


class Image(object):
    """
    Image with only one channel
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrix = [[0 for i in range(width)] for j in range(height)]

    @classmethod
    def create_image(cls, matrix):
        img = cls(width=len(matrix[0]), height=len(matrix))
        for i in range(img.width):
            for j in range(img.height):
                img.matrix[j][i] = matrix[j][i]
        return img

    def __getitem__(self, item):
        return self.matrix[item]


def convolve(image, kernal):
    # Use symmetrical boundary conditions to fill pixels outside image with zeros
    expand_length = int(math.floor(kernal.height / 2.0))
    expanded_image = Image(height=image.height + 2 * expand_length, width=image.width + 2 * expand_length)
    new_image = Image(height=image.height, width=image.width)
    for i in range(image.height):
        for j in range(image.height):
            expanded_image[i + expand_length][j + expand_length] = image[i][j]
    for i in range(image.height):
        for j in range(image.height):
            pixel = 0
            for m in range(kernal.height):
                for n in range(kernal.width):
                    pixel += int(expanded_image[i - expand_length + m][j - expand_length + n] * kernal[m][n])
            new_image[i][j] = pixel

    return new_image


class ConvolveGate(Gate):
    def __init__(self):
        super(ConvolveGate, self).__init__()

    def _forward(self, *units):
        pass

    def _backward(self):
        pass


class ConvolutionalNeuralNetwork(Network):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

    def _forward(self, *units):
        pass

    def _backward(self):
        pass

    @property
    def weights(self):
        pass

    @property
    def weights_without_bias(self):
        pass


def how_to_understand_convolution():
    img = Image.create_image([
        [25, 85, 22],
        [38, 21, 31],
        [40, 35, 33],
    ])
    plt.imshow(img.matrix)
    plt.show()
    # Average the noise
    descriptor = Image.create_image([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ])
    new_image = convolve(img, descriptor)
    plt.imshow(new_image.matrix)
    plt.show()


if __name__ == '__main__':
    how_to_understand_convolution()
