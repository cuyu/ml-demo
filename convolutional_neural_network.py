#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Note:
    1.
"""
import math
import matplotlib.pyplot as plt
from neural_network import Unit, Gate, Network, Neuron, NeuralNetwork


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

    def __init__(self, input_shape, kernel_shape, kernel_number, stride=1):
        """
        :param input_shape: a 3-dimension list which meaning is [width, height, depth] of the inputs
        :param kernel_shape: a 3-dimension list which meaning is [width, height, depth] of the convolution kernel
        :param kernel_number: determine how many kernel will be used in this layer
        :param stride: the interval between two convolution window
        """
        super(ConvReluLayer, self).__init__()
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.kernel_number = kernel_number
        self.stride = stride
        self.output_shape = [int(input_shape[0] / self.stride), int(input_shape[1] / self.stride), kernel_number]
        # We use only one <Neuron> instance for a kernel as all the neurons share the same weights
        self.common_neurons = [
            Neuron(feature_length=kernel_shape[0] * kernel_shape[1] * kernel_shape[2], activation_function='relu') for _
            in range(kernel_number)]

    def _forward(self, unit_cube):
        """
        :param unit_cube: a <UnitCube> instance
        :return: a <UnitCube> instance
        """
        utop = UnitCube(width=self.output_shape[0], height=self.output_shape[1], depth=self.kernel_number)
        # The width/height of the padding that we need to fulfill with zeros
        expand_width = int(math.floor(self.kernel_shape[0] / 2.0))
        expand_height = int(math.floor(self.kernel_shape[1] / 2.0))
        for d in range(self.kernel_number):
            neuron = self.common_neurons[d]
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[0]):
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


class CNNStructure(object):
    _SUPPORT_LAYERS = frozenset(['CONV', 'POOL', 'FC'])

    def __init__(self):
        self.layers = []

    def add_layer(self, layer_type, layer_options):
        """
        :param layer_type: a string
        :param layer_options: a dict of valid arguments for corresponding layer. For 'FC' layer, the layer_option should
            be {'neuron_number': xxx}.
        """
        assert layer_type in self._SUPPORT_LAYERS
        if layer_type == 'CONV':
            self.layers.append(ConvReluLayer(**layer_options))
        elif layer_type == 'POOL':
            last_layer = self.layers[-1]
            assert isinstance(last_layer, ConvReluLayer), 'Only support pooling after conv layer'
            self.layers.append(MaxPoolingLayer(input_shape=last_layer.output_shape, **layer_options))
        elif layer_type == 'FC':
            # todo
            dnn_structure = []


class ConvolutionalNeuralNetwork(Network):
    def __init__(self, network_structure):
        """
        :param network_structure: a <CNNStructure> instance
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.network_structure = network_structure
        self.layers = network_structure.layers

    def _forward(self, *units):
        for l in self.layers:
            utop = l.forward()
        return utop

    def _backward(self):
        for l in reversed(self.layers):
            l.backward()

    @property
    def weights(self):
        w = []
        for n in self.layers:
            w += n.weights
        return w

    @property
    def weights_without_bias(self):
        w = []
        for n in self.layers:
            w += n.weights_without_bias
        return w


if __name__ == '__main__':
    from mnist import MNIST

    mndata = MNIST('./data', gz=True)
    images, labels = mndata.load_testing()
    img0 = images[0]
    img0_cube = UnitCube(width=28, height=28, depth=1)
    for i in range(28):
        for j in range(28):
            img0_cube[0][i][j].value = img0[i * 28 + j]
    conv_layer = ConvReluLayer(input_shape=[28, 28, 1], kernel_shape=[3, 3, 1], kernel_number=3)
    conv_layer.forward(img0_cube)
    conv_layer.backward()
    conv_layer.utop.imshow(0)
    max_pool_layer = MaxPoolingLayer(input_shape=[28, 28, 1], pooling_shape=[2, 2])
    max_pool_layer.forward(img0_cube)
    max_pool_layer.backward()
