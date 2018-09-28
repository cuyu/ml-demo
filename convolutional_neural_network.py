#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note:
    1.
"""
import math
import matplotlib.pyplot as plt
from neural_network import Unit, Gate, Network, Neuron


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
    def __init__(self, input_shape, pooling_shape, stride=None):
        """
        :param input_shape: a 3-dimension list which meaning is [width, height, depth] of the inputs
        :param pooling_shape: a 2-dimension list which meaning is [width, height] of the pooling
        :param stride: the interval between two pooling window, by default will use the width of the pooling shape
        """
        super(MaxPoolingLayer, self).__init__()
        self.input_shape = input_shape
        self.pooling_shape = pooling_shape
        self.stride = stride if stride else pooling_shape[0]
        # fixme: the pool width/height should consider stride
        self.pool_width_number = int(input_shape[0] / pooling_shape[0])
        self.pool_height_number = int(input_shape[1] / pooling_shape[1])
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
                        *unit_cube.range(width_range=range(j * self.stride, j * self.stride + self.pooling_shape[0]),
                                         height_range=range(i * self.stride, i * self.stride + self.pooling_shape[1]),
                                         depth_range=[d]))
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


class ConvReluLayer(Network):
    """
    A convolution layer follows with a ReLu layer.
    (We put the two layers together as they always used together. It is meaningless to use CONV -> CONV -> RELU,
     because the adjacent CONV layers can be merged into one CONV layer)
    """

    def __init__(self, kernel_shape, kernel_number, stride=1):
        """
        :param kernel_shape: a 3-dimension list which meaning is [width, height, depth] of the convolution kernel
        :param kernel_number: determine how many kernel will be used in this layer
        :param stride: the interval between two convolution window
        """
        super(ConvReluLayer, self).__init__()
        self.kernel_shape = kernel_shape
        self.kernel_number = kernel_number
        self.stride = stride
        # We use only one <Neuron> instance for a kernel as all the neurons share the same weights
        self.common_neurons = [
            Neuron(feature_length=kernel_shape[0] * kernel_shape[1] * kernel_shape[2], activation_function='relu') for _
            in range(kernel_number)]

    def _forward(self, unit_cube):
        """
        :param unit_cube: a <UnitCube> instance
        :return: a <UnitCube> instance
        """
        output_width = int(unit_cube.width / self.stride)
        output_height = int(unit_cube.height / self.stride)
        utop = UnitCube(width=output_width, height=output_height, depth=self.kernel_number)
        # The width/height of the padding that we need to fulfill with zeros
        expand_width = int(math.floor(self.kernel_shape[0] / 2.0))
        expand_height = int(math.floor(self.kernel_shape[1] / 2.0))
        for d in range(self.kernel_number):
            neuron = self.common_neurons[d]
            for i in range(output_height):
                for j in range(output_width):
                    # For a 3 * 3 kernel, for (0, 0), it should pick width_range [-1, 2)
                    units = unit_cube.range(width_range=range(j * self.stride - expand_width,
                                                              j * self.stride - expand_width + self.kernel_shape[0]),
                                            height_range=range(i * self.stride - expand_height,
                                                               i * self.stride - expand_height + self.kernel_shape[1]),
                                            depth_range=range(unit_cube.depth))
                    assert len(units) == neuron.feature_length
                    utop[d][i][j] = neuron.forward(*units)

        return utop

    def _backward(self):
        for d in reversed(range(self.kernel_number)):
            neuron = self.common_neurons[d]
            for i in reversed(range(self.utop.height)):
                for j in reversed(range(self.utop.width)):
                    neuron.backward()

    @property
    def weights(self):
        w = []
        for n in self.common_neurons:
            w += n.weights
        return w

    @property
    def weights_without_bias(self):
        w = []
        for n in self.common_neurons:
            w += n.weights_without_bias
        return w


class UnitCube(object):
    """
    A group of <Unit> instance, distributed in a cube
    """

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.cube = [[[Unit(0) for i in range(width)] for j in range(height)] for k in range(depth)]

    def range(self, width_range, height_range, depth_range, padding='zero'):
        """
        :param width_range, height_range, depth_range: a list of indexes
        :param padding: choose how to fill the padding outside range
        :return: A list of <Unit>
        """
        result = []
        for d in depth_range:
            for i in height_range:
                for j in width_range:
                    try:
                        result.append(self.cube[d][i][j])
                    except IndexError:
                        if padding == 'zero':
                            result.append(Unit(0))
                        else:
                            raise Exception('Padding method not supported!')
        return result

    def imshow(self, depth):
        """
        Plot image of given depth
        """
        layer = self.cube[depth]
        img = []
        for i in range(self.height):
            img.append([layer[i][j].value for j in range(self.width)])
        plt.imshow(img, cmap='gray')
        plt.show()

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

    mndata = MNIST('./data', gz=True)
    images, labels = mndata.load_testing()
    img0 = images[0]
    img0_cube = UnitCube(width=28, height=28, depth=1)
    for i in range(28):
        for j in range(28):
            img0_cube[0][i][j].value = img0[i * 28 + j]
    conv_layer = ConvReluLayer(kernel_shape=[3, 3, 1], kernel_number=3)
    conv_layer.forward(img0_cube)
    conv_layer.backward()
    conv_layer.utop.imshow(0)
    max_pool_layer = MaxPoolingLayer(input_shape=[28, 28, 1], pooling_shape=[2, 2])
    max_pool_layer.forward(img0_cube)
    max_pool_layer.backward()
    how_to_understand_convolution()
