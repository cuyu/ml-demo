#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implement a neural network, using modular.
Support input features with arbitrary dimensions
"""
import random
from abc import abstractmethod
import math


class Unit(object):
    def __init__(self, value, gradient=0):
        self.value = value
        self.gradient = gradient


class Gate(object):
    """
    The base class of all gates
    """

    def __init__(self):
        self.utop = None
        # Will be a tuple after first forward function
        self.units = None
        self._units_history = []
        self._utop_history = []

    @property
    def value(self):
        return self.utop.value

    @value.setter
    def value(self, new_value):
        self.utop.value = new_value

    @property
    def gradient(self):
        return self.utop.gradient

    @gradient.setter
    def gradient(self, new_value):
        self.utop.gradient = new_value

    @abstractmethod
    def _forward(self, *units):
        """
        :param units: <Unit> instances
        :return: utop, a <Unit> instance
        """
        raise NotImplementedError

    def forward(self, *units):
        """
        :param units: <Unit> instances
        :return utop
        Run the forward process, calculate the utop (a <Unit> instance) as output.
        An example of a gate with two inputs is like below:

        u0 --- [      ]
               [ Gate ] --- utop
        u1 --- [      ]

        Note: the method should not be overrode! Override _forward method instead.
        """
        self.units = units
        self.utop = self._forward(*units)
        self._units_history.append(units)
        self._utop_history.append(self.utop)
        return self.utop

    def clean_history(self):
        self._units_history = []
        self._utop_history = []

    @abstractmethod
    def _backward(self):
        """
        :return: None
        """
        raise NotImplementedError

    def backward(self):
        """
        Run the backward process to update the gradient of each unit in the gate

        Note: the method should not be overrode! Override _backward method instead.
        """
        self.units = self._units_history.pop()
        self._backward()
        # We must set the utop to previous state immediately, because the utop could be other gate's input unit
        # And other gate's backward could be called before this gate's backward
        self._utop_history.pop()
        if self._utop_history:
            self.utop = self._utop_history[-1]

    def set_utop_gradient(self, gradient):
        """
        Must be called before backward function for the final <Gate> instance
        todo: how do we check if this method is called as the gradient default is 0
        """
        self.utop.gradient = gradient


class AddGate(Gate):
    def __init__(self):
        super(AddGate, self).__init__()

    def _forward(self, *units):
        return Unit(sum([u.value for u in units]))

    def _backward(self):
        for u in self.units:
            u.gradient += 1 * self.utop.gradient


class MultiplyGate(Gate):
    def __init__(self):
        super(MultiplyGate, self).__init__()

    def _forward(self, u0, u1):
        self.utop = Unit(self.units[0].value * self.units[1].value)
        return self.utop

    def _backward(self):
        """
        Use the chain rule, assume f as the final output:
        d(f)/d(u0) = d(f)/d(utop) * d(utop)/d(u0)
                   = utop.gradient * u1.value
        """
        self.units[0].gradient += self.units[1].value * self.utop.gradient
        self.units[1].gradient += self.units[0].value * self.utop.gradient


class SigmoidGate(Gate):
    def __init__(self):
        super(SigmoidGate, self).__init__()

    def _forward(self, u0):
        self.utop = Unit(1 / (1 + math.exp(-u0.value)))
        return self.utop

    def _backward(self):
        self.units[0].gradient += self.units[0].value * (1 - self.units[0].value) * self.utop.gradient


class ReLUGate(Gate):
    def __init__(self):
        super(ReLUGate, self).__init__()

    def _forward(self, u0):
        self.utop = Unit(max(0, self.units[0].value))
        return self.utop

    def _backward(self):
        """
        Here, we define the derivative at x=0 to 0
        Refer to https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
        """
        if self.units[0].value > 0:
            self.units[0].gradient += 1 * self.utop.gradient
        else:
            self.units[0].gradient += 0 * self.utop.gradient


class Network(Gate):
    """
    Base class of networks
    """

    def __init__(self):
        super(Network, self).__init__()

    @abstractmethod
    def _forward(self, *units):
        raise NotImplementedError

    @abstractmethod
    def _backward(self):
        raise NotImplementedError

    def pull_weights(self, learning_rate):
        """
        Adjust all the weights according to the gradients
        Should be called after forward and backward process
        """
        for w in self.weights:
            w.value += learning_rate * w.gradient
        # Reset all the weights' gradient to 0
        # We will not reset all other units' gradient, because all other units should be initialized in next training
        # round, and the init value of gradient is 0
        for w in self.weights:
            w.gradient = 0

    @property
    @abstractmethod
    def weights(self):
        """
        :return: All the weights used inside the network, each weight is a <Unit> instance
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def weights_without_bias(self):
        """
        :return: The weights related to input feature, i.e. exclude bias weights from all weights
                 Used to calculate Regularization in loss function
        """
        raise NotImplementedError


class LinearNetwork(Network):
    """
    A LinearNetwork:
        f(X) = ∑w_i * x_i + c
    Where, X is the input vector, x_i is one dimension of X, w_i is the weight for each x_i, c is a constant

    So we need n MultiplyGate (n is the feature length) and one AddGate here
    """

    def __init__(self, feature_length):
        """
        :param feature_length: dimension number of the feature
        """
        super(LinearNetwork, self).__init__()
        self.feature_length = feature_length
        # [Note]: we should use different init values for weights here
        # If we use the same init values (e.g. 1.0), then all the Neurons will be all the same no matter how many rounds
        # we trained (For each round, the output of each neuron will be same, so the gradient will be same too)
        self.W = [Unit(random.random()) for _ in range(feature_length)]
        self.c = Unit(1.0)
        self.multi_gates = [MultiplyGate() for _ in range(feature_length)]
        self.add_gate = AddGate()

    def _forward(self, *units):
        assert len(units) == self.feature_length, "The input feature dimension should be consistent!"
        for i in range(len(units)):
            self.multi_gates[i].forward(self.W[i], units[i])
        utop = self.add_gate.forward(*self.multi_gates, self.c)
        return utop

    def _backward(self):
        self.add_gate.backward()
        for gate in reversed(self.multi_gates):
            gate.backward()

    @property
    def weights(self):
        return self.W + [self.c]

    @property
    def weights_without_bias(self):
        return self.W


class Neuron(Network):
    """
    a Neuron of NeuralNetwork
    For input (x, y), the formula is:
        f(x, y) = max(0, a * x + b * y + c)
    We can just think as it put the output of a <LinearNetwork> into a <ReLU> Gate
    """

    def __init__(self, feature_length):
        super(Neuron, self).__init__()
        self.linear_network = LinearNetwork(feature_length)
        self.relu_gate = ReLUGate()

    def _forward(self, *units):
        self.linear_network.forward(*units)
        self.utop = self.relu_gate.forward(self.linear_network)
        return self.utop

    def _backward(self):
        self.relu_gate.backward()
        self.linear_network.backward()

    @property
    def weights(self):
        return self.linear_network.weights

    @property
    def weights_without_bias(self):
        return self.linear_network.weights_without_bias


class SingleLayerNeuralNetwork(Network):
    """
    A neural network with only output layer which consist of two <Neuron>:
    x - neuron1
      X        > f(x, y)
    y - neuron2

    The formula is:
        f(x, y) = a1 * n1 + a2 * n2 + d
    where n1, n2 is the output of <Neuron>, just as simple as apply the LinearNetwork to the <Neuron>
    """

    def __init__(self, feature_length, neuron_number):
        super(SingleLayerNeuralNetwork, self).__init__()
        self.feature_length = feature_length
        self.neuron_number = neuron_number
        self.neurons = [Neuron(feature_length) for _ in range(neuron_number)]
        self.linear_network = LinearNetwork(neuron_number)

    def _forward(self, *units):
        assert len(units) == self.feature_length, "The input feature dimension should be consistent!"
        for n in self.neurons:
            n.forward(*units)
        self.utop = self.linear_network.forward(*self.neurons)
        return self.utop

    def _backward(self):
        self.linear_network.backward()
        for n in reversed(self.neurons):
            n.backward()

    @property
    def weights(self):
        w = []
        for n in self.neurons:
            w += n.weights
        return w + self.linear_network.weights

    @property
    def weights_without_bias(self):
        w = []
        for n in self.neurons:
            w += n.weights_without_bias
        return w + self.linear_network.weights_without_bias


class NeuralNetwork(Network):
    """
    A neural network with one hidden layer and one output layer:
    x - n1 - n3
      X    X    > f(x, y)
    y - n2 - n4

    Where, the n1, n2 are <Neuron> in hidden layer, n3, n4 are <Neuron> in output layer, x, y are inputs
    """

    def __init__(self, feature_length, network_structure):
        """
        :param feature_length: a Int number
        :param network_structure: a list, where each number means the neurons number for each layer.
        e.g. [4, 8, 16] means two hidden layers which has 4 and 8 neurons for each, and one output layer with 16 neurons
        """
        super(NeuralNetwork, self).__init__()
        self.feature_length = feature_length
        self.network_structure = network_structure
        # The first layer's feature length is the actual feature length
        self.layers = [SingleLayerNeuralNetwork(feature_length=feature_length, neuron_number=network_structure[0])]
        # The other layer's feature length is the neuron number of its former layer
        for i in range(1, len(network_structure)):
            self.layers.append(
                SingleLayerNeuralNetwork(feature_length=network_structure[i - 1], neuron_number=network_structure[i]))

    def _forward(self, *units):
        assert len(units) == self.feature_length, "The input feature dimension should be consistent!"
        utop = self.layers[0].forward(*units)
        for i in range(1, len(self.layers)):
            utop = self.layers[i].forward(*self.layers[i - 1].neurons)
        return utop

    def _backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    @property
    def weights(self):
        w = []
        for layer in self.layers:
            w += layer.weights
        return w

    @property
    def weights_without_bias(self):
        w = []
        for layer in self.layers:
            w += layer.weights_without_bias
        return w


class LossNetwork(Network):
    """
    The network to calculate the loss of the classification

    For a simple linear classifier, the formula:
        L=∑max(0,−y_i*(w_0*x_i0+w_1*x_i1+w_2)+1)+α*(w_0*w_0+w_1*w_1)
    Where, the x_i0,x_i1 are the input feature, y_i is the label, w_0, w_1 are the weights related to features
    (Not include the bias, i.e. a, b in above linear network, but not include c)

    For general classifier, the formula:
        L=∑max(0,−y_i*f(X_i)+1)+α*∑w_i*w_i
    Where, the X_i is 1*N feature vector, f(X_i) is the output (utop.value) of given Network
    """

    def __init__(self, network, data_set, alpha=0.1):
        """
        :param network: a <Network> instance
        :param data_set: a list of tuple, the first of the tuple is the feature vector, the second is the label
        """
        super(LossNetwork, self).__init__()
        self.network = network
        self.data_set = data_set
        self.alpha = alpha  # Regularization strength
        feature_number = len(data_set)
        # Multiply in `−y_i*f(X_i)+1`
        self.multi_gates_l = [MultiplyGate() for _ in range(feature_number)]
        # Add in `−y_i*f(X_i)+1`
        self.add_gates_l = [AddGate() for _ in range(feature_number)]
        # ReLU in `max(0,−y_i*f(X_i)+1)`
        self.relu_gates_l = [ReLUGate() for _ in range(feature_number)]
        # Add of the ∑ in `∑max(0,−y_i*f(X_i)+1)`
        self.add_gates_sigma_l = AddGate()
        weight_number = len(self.network.weights_without_bias)
        # Multiply in `w_i*w_i`
        self.multi_gates_r = [MultiplyGate() for _ in range(weight_number)]
        # Add of ∑ in `∑w_i*w_i`
        self.add_gates_sigma_r = AddGate()
        # Multiply alpha in `α*∑w_i*w_i`
        self.multi_gate_alpha = MultiplyGate()
        # The final add
        self.add_gate_final = AddGate()

    def _forward(self):
        # Calculate ∑max(0,−y_i*f(X_i)+1)
        for i in range(len(self.relu_gates_l)):
            feature = self.data_set[i][0]
            label = self.data_set[i][1]
            self.network.forward(*[Unit(x) for x in feature])
            self.multi_gates_l[i].forward(Unit(-label), self.network)
            self.add_gates_l[i].forward(self.multi_gates_l[i], Unit(1.0))
            self.relu_gates_l[i].forward(self.add_gates_l[i])
        self.add_gates_sigma_l.forward(*self.relu_gates_l)
        # Calculate α*∑w_i*w_i
        for i in range(len(self.multi_gates_r)):
            self.multi_gates_r[i].forward(self.network.weights_without_bias[i], self.network.weights_without_bias[i])
        self.add_gates_sigma_r.forward(*self.multi_gates_r)
        self.multi_gate_alpha.forward(Unit(self.alpha), self.add_gates_sigma_r)
        utop = self.add_gate_final.forward(self.add_gates_sigma_l, self.multi_gate_alpha)
        return utop

    def _backward(self):
        self.add_gate_final.backward()
        self.multi_gate_alpha.backward()
        self.add_gates_sigma_r.backward()
        for gate in reversed(self.multi_gates_r):
            gate.backward()
        self.add_gates_sigma_l.backward()
        # df/dw_0 = 2*α*w_0 - y_i * x_i0  OR  df/dw_0 = 2*α*w_0
        # Take the below input as example:
        # ([-0.1, -1.0], -1),
        # ([-1.0, 1.1], -1),
        # Then the gradient of w_0 and w_1 for each loop backward should be:
        # loop(i=1): 0.8, -1.3
        # loop(i=0): 0.8 + 0.1 = 0.9, -1.3 + 1 = -0.3
        for i in reversed(range(len(self.relu_gates_l))):
            self.relu_gates_l[i].backward()
            self.add_gates_l[i].backward()
            self.multi_gates_l[i].backward()
            self.network.backward()

    @property
    def weights(self):
        return self.network.weights

    @property
    def weights_without_bias(self):
        return self.network.weights_without_bias


class BasicClassifier(object):
    """
    The base class of classifiers
    """

    def __init__(self, network):
        """
        :param network: A <Network> instance
        """
        self.network = network
        self._loss_by_step = []

    def simple_train(self, data_set, learning_rate=0.01, steps=100):
        """
        Train the classifier only using single feature and label
        :param data_set: a list of tuple, the first of the tuple is the feature vector, the second is the label
        :param learning_rate: the learning rate
        :param steps: how many rounds for training
        """
        for _ in range(steps):
            for feature, label in data_set:
                utop = self.network.forward(*[Unit(k) for k in feature])
                if label > 0 and utop.value < 1:
                    pull = 1
                elif label < 0 and utop.value > -1:
                    pull = -1
                else:
                    pull = 0
                # Set the gradient of final unit and then backward to get the direction (gradient) of corresponding parameters
                # We can also set the pull (i.e. gradient) more/less than 1 to make the adjust more efficient
                self.network.set_utop_gradient(pull)
                self.network.backward()
                self.network.pull_weights(learning_rate)

    def train(self, data_set, learning_rate=0.01, steps=100):
        """
        Train the classifier using loss function (with all the features and labels)
        """
        loss_network = LossNetwork(self.network, data_set)
        for _ in range(steps):
            loss_network.forward()
            self._loss_by_step.append(loss_network.value)
            loss_network.set_utop_gradient(-1)
            loss_network.backward()
            loss_network.pull_weights(learning_rate)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self._loss_by_step)
        plt.ylabel('loss')
        plt.show()

    def predict(self, feature):
        """
        :param feature: A list of number
        """
        predicted = self.network.forward(*[Unit(i) for i in feature]).value
        # Prevent memory leak
        self.network.clean_history()
        return predicted


class LinearClassifier(BasicClassifier):
    def __init__(self, feature_length):
        network = LinearNetwork(feature_length)
        super(LinearClassifier, self).__init__(network)


class NeuralNetworkClassifier(BasicClassifier):
    def __init__(self, feature_length, network_structure):
        network = NeuralNetwork(feature_length, network_structure)
        super(NeuralNetworkClassifier, self).__init__(network)


if __name__ == '__main__':
    data_set = [
        ([1.2, 0.7], -1),
        ([-0.3, -0.5], -1),
        ([3.0, 0.1], 1),
        ([-0.1, -1.0], -1),
        ([-1.0, 1.1], -1),
        ([2.1, -3.0], 1),
        ([1.1, -1.0], 1),
    ]
    # data_set = [
    #     ([1.2], 1),
    #     ([-0.3], -1),
    #     ([2.1], 1),
    #     ([-1.0], -1),
    #     ([0.8], -1),
    #     ([-3.0], 1),
    #     ([-2.0], 1),
    # ]
    classifier = NeuralNetworkClassifier(feature_length=2, network_structure=[4, 8])
    # classifier.simple_train(data_set)
    classifier.train(data_set, learning_rate=0.01, steps=200)
    classifier.plot_loss()
    print('---')
    import matplotlib.pyplot as plt

    for x in range(-30, 30):
        for y in range(-30, 30):
            _x = x * 0.1
            _y = y * 0.1
            label = classifier.predict([_x, _y])
            color = '#a1d5ed' if label > 0 else '#efaabd'
            plt.plot(_x, _y, color, marker='*')

    for feature, label in data_set:
        print(classifier.predict(feature))
        color = 'b' if label > 0 else 'r'
        plt.plot(*feature, color + 'o')
    print([u.value for u in classifier.network.weights])

    plt.show()
