#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note:
    1.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from neural_network import Unit, Gate, Network


class MaxGate(Gate):
    def __init__(self):
        super(MaxGate, self).__init__()
        self._max_unit = None

    def _forward(self, *units):
        max_unit = units[0]
        for u in units[1:]:
            if u.value > max_unit.value:
                max_unit = u
        self._max_unit = max_unit
        return Unit(max_unit.value)

    def _backward(self):
        self._max_unit.gradient += self.utop.gradient


class MaxPoolingLayer(Network):
    def __init__(self, input_shape, pooling_shape, stride=1):
        """
        :param input_shape: a 3-dimension list which meaning is [width, height, depth] of the inputs
        :param pooling_shape: a 2-dimension list which meaning is [width, height] of the pooling
        :param stride: the interval between two pooling window
        """
        super(MaxPoolingLayer, self).__init__()
        self.input_shape = input_shape
        self.pooling_shape = pooling_shape
        self.stride = stride
        self.pool_width_number = input_shape[0] / pooling_shape[0]
        self.pool_height_number = input_shape[1] / pooling_shape[1]
        self.max_gates = [[MaxGate() for i in range(self.pool_width_number)] for j in
                          range(self.pool_height_number)]

    def _forward(self, unit_cube):
        """
        :param unit_cube: a <UnitCube> instance
        :return: a <UnitCube> instance
        """
        height, width, depth = self.input_shape
        utop = UnitCube(self.pool_width_number, self.pool_height_number, depth)
        for d in range(depth):
            for i in range(self.pool_height_number):
                for j in range(self.pool_width_number):
                    utop[d][i][j] = self.max_gates[i][j].forward(
                        unit_cube.range(width=range(i, i + self.pooling_shape[0]),
                                        height=range(j, j + self.pooling_shape[1]),
                                        depth=d))
        return utop

    def _backward(self):
        # The sequence does not matters here
        for i in range(self.pool_height_number):
            for j in range(self.pool_width_number):
                self.max_gates[i][j].backward()

    @property
    def weights(self):
        return []

    @property
    def weights_without_bias(self):
        return []


class UnitCube(object):
    """
    A group of <Unit> instance, distributed in a cube
    """

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.cube = [[[Unit(0) for i in range(width)] for j in range(height)] for k in range(depth)]

    def range(self, width, height, depth):
        return self.cube

    def __getitem__(self, item):
        """
        UnitCube[i] => get depth i
        UnitCube[i][j] => get depth i, height j
        UnitCube[i][j][k] => get depth i, height j, width k
        """
        return self.cube[item]


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
                    pixel += expanded_image[i - expand_length + m][j - expand_length + n] * kernal[m][n]
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
    from mnist import MNIST

    mndata = MNIST('./data')
    mndata.gz = True
    images, labels = mndata.load_training()
    how_to_understand_convolution()
