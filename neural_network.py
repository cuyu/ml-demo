#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implement a simple neural network, support any dimensions feature.
"""
from abc import abstractmethod
import math


class Unit(object):
    def __init__(self, value, gradient=None):
        self.value = value
        self.gradient = gradient


class Gate(object):
    """
    The base class of all gates
    """

    def __init__(self):
        self.utop = None

    def _set_utop_gradient(self):
        """
        Should only called by backward function
        Set the utop gradient to 1.0 if it is the start of backward chain
        """
        if self.utop and self.utop.gradient is None:
            self.utop.gradient = 1.0

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
    def forward(self, *units):
        """
        :param units: <Unit> instances
        :return utop
        Run the forward process, calculate the utop (a <Unit> instance) as output.
        An example of a gate with two inputs is like below:

        u0 --- [      ]
               [ Gate ] --- utop
        u1 --- [      ]
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        """
        Run the backward process to update the gradient of each unit in the gate
        """
        raise NotImplementedError


class AddGate(Gate):
    def __init__(self):
        super(AddGate, self).__init__()
        self.u0 = None
        self.u1 = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(self.u0.value + self.u1.value)
        return self.utop

    def backward(self):
        self._set_utop_gradient()
        self.u0.gradient = 1 * self.utop.gradient
        self.u1.gradient = 1 * self.utop.gradient


class MultiplyGate(Gate):
    def __init__(self):
        super(MultiplyGate, self).__init__()
        self.u0 = None
        self.u1 = None

    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(self.u0.value * self.u1.value)
        return self.utop

    def backward(self):
        """
        Use the chain rule, assume f as the final output:
        d(f)/d(u0) = d(f)/d(utop) * d(utop)/d(u0)
                   = utop.gradient * u1.value
        """
        self._set_utop_gradient()
        self.u0.gradient = self.u1.value * self.utop.gradient
        self.u1.gradient = self.u0.value * self.utop.gradient


class SigmoidGate(Gate):
    def __init__(self):
        super(SigmoidGate, self).__init__()
        self.u0 = None

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(1 / (1 + math.exp(-self.u0.value)))
        return self.utop

    def backward(self):
        self._set_utop_gradient()
        self.u0.gradient = self.u0.value * (1 - self.u0.value) * self.utop.gradient


class ReLUGate(Gate):
    def __init__(self):
        super(ReLUGate, self).__init__()
        self.u0 = None

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(max(0, self.u0.value))
        return self.utop

    def backward(self):
        """
        Here, we define the derivative at x=0 to 0
        Refer to https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
        """
        self._set_utop_gradient()
        if self.u0.value > 0:
            self.u0.gradient = 1 * self.utop.gradient
        else:
            self.u0.gradient = 0 * self.utop.gradient


class Network(Gate):
    """
    Base class of networks
    """

    def __init__(self):
        super(Network, self).__init__()

    @abstractmethod
    def forward(self, *units):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

    def pull_weights(self, learning_rate):
        """
        Adjust all the weights according to the gradients
        Should be called after forward and backward process
        """
        for w in self.weights:
            w.value += learning_rate * w.gradient

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
    A LinearNetwork: it takes 5 Units (x,y,a,b,c) and outputs a single Unit:
        f(x, y) = a * x + b * y + c
    So we need two MultiplyGate and two AddGate here

    From outside of the network, we can assume the whole network as a gate, which has 5 inputs and 1 output
    So we inherit the <Gate> class here
    """

    def __init__(self, feature_length):
        super(LinearNetwork, self).__init__()
        assert feature_length >= 2  # todo: support feature_length = 1
        self._feature_length = feature_length
        self._weights = [Unit(1.0) for _ in range(feature_length + 1)]
        self.multi_gates = [MultiplyGate() for _ in range(feature_length)]
        self.add_gates = [AddGate() for _ in range(feature_length)]

    def forward(self, *feature):
        """
        :param feature: <Unit> instances
        """
        assert len(feature) == self._feature_length
        for i in range(len(feature)):
            self.multi_gates[i].forward(self._weights[i], feature[i])
        self.add_gates[0].forward(self.multi_gates[0], self._weights[-1])
        for i in range(1, len(feature)):
            self.utop = self.add_gates[i].forward(self.add_gates[i - 1], self.multi_gates[i])
        return self.utop

    def backward(self):
        self._set_utop_gradient()
        # The sequence matters here!
        for gate in reversed(self.add_gates):
            gate.backward()
        for gate in reversed(self.multi_gates):
            gate.backward()

    @property
    def weights(self):
        return self._weights

    @property
    def weights_without_bias(self):
        return self._weights[:-1]


class SingleNeuralNetwork(Network):
    """
    SingleNeuralNetwork, a.k.a. a Neuro of NeuralNetwork
    The formula is:
        f(x, y) = max(0, a * x + b * y + c)
    We can just think as it put the output of a <LinearNetwork> into a <ReLU> Gate
    """

    def __init__(self, feature_length):
        super(SingleNeuralNetwork, self).__init__()
        self.linear_network = LinearNetwork(feature_length)
        self.relu_gate = ReLUGate()

    def forward(self, *feature):
        self.linear_network.forward(*feature)
        self.utop = self.relu_gate.forward(self.linear_network)
        return self.utop

    def backward(self):
        self._set_utop_gradient()
        self.relu_gate.backward()
        self.linear_network.backward()

    @property
    def weights(self):
        return self.linear_network.weights

    @property
    def weights_without_bias(self):
        return self.linear_network.weights_without_bias


class NeuralNetwork(Network):
    """
    A neural network consist of two <SingleNeuralNetwork>
    The formula is:
        f(x, y) = a1 * n1 + a2 * n2 + d
    where n1, n2 is the output of <SingleNeuralNetwork>, just as simple as apply the LinearNetwork to the <SingleNeuralNetwork>
    """

    def __init__(self, feature_length, neuro_number):
        super(NeuralNetwork, self).__init__()
        self.neuros = [SingleNeuralNetwork(feature_length) for _ in range(neuro_number)]
        self.linear_network = LinearNetwork(neuro_number)

    def forward(self, *feature):
        for neuro in self.neuros:
            neuro.forward(*feature)
        self.utop = self.linear_network.forward(*self.neuros)
        return self.utop

    def backward(self):
        self._set_utop_gradient()
        self.linear_network.backward()
        for neuro in reversed(self.neuros):
            neuro.backward()

    @property
    def weights(self):
        _weights = []
        for neuro in self.neuros:
            _weights += neuro.weights
        _weights += self.linear_network.weights
        return _weights

    @property
    def weights_without_bias(self):
        _weights = []
        for neuro in self.neuros:
            _weights += neuro.weights_without_bias
        _weights += self.linear_network.weights_without_bias
        return _weights


class BasicClassifier(object):
    """
    The base class of classifiers
    """

    def __init__(self, network):
        """
        :param network: A <Network> instance
        """
        self.network = network

    def train(self, data_set, learning_rate=0.01, steps=100):
        """
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
                self.network.gradient = pull
                self.network.backward()
                self.network.pull_weights(learning_rate)

    def predict(self, x, y):
        return self.network.forward(Unit(x), Unit(y)).value


class LinearClassifier(BasicClassifier):
    def __init__(self):
        network = LinearNetwork(2)  # todo: remove the hard code here
        super(LinearClassifier, self).__init__(network)


class NeuralNetworkClassifier(BasicClassifier):
    def __init__(self):
        network = NeuralNetwork(2, 3)  # todo: remove the hard code here
        super(NeuralNetworkClassifier, self).__init__(network)


class AdvancedClassifier(object):
    """
    The network to calculate the cost of the classification

    For a simple linear classifier, the formula:
        L=∑max(0,−y_i*(w_0*x_i0+w_1*x_i1+w_2)+1)+α(w_0*w_0+w_1*w_1)
    Where, the x_i0,x_i1 are the input feature, y_i is the label, w_0, w_1 are the weights related to features
    (Not include the bias, i.e. a, b in above linear network, but not include c)

    For general classifier, the formula:
        L=∑max(0,−y_i*f(X_i)+1)+α*∑w_i*w_i
    Where, the X_i is 1*N feature vector, f(X_i) is the output (utop.value) of given Network
    """

    def __init__(self, network):
        self.network = network
        self.alpha = 0.1  # Regularization strength

    def train(self, data_set, learning_rate=0.01, steps=100):
        feature_number = len(data_set)
        multi_gates = [MultiplyGate() for _ in range(feature_number)]
        add_gates = [AddGate() for _ in range(feature_number * 3)]
        weight_number = len(self.network.weights_without_bias)

    def forward(self):
        self.mutli_gates

    def backward(self):
        self._set_utop_gradient()

    def pull_parameters(self, learning_rate):
        pass


if __name__ == '__main__':
    data_set = [
        ([1.2, 0.7], 1),
        ([-0.3, -0.5], -1),
        ([3.0, 0.1], 1),
        ([-0.1, -1.0], -1),
        ([-1.0, 1.1], -1),
        ([2.1, -3.0], 1),
    ]
    classifier = NeuralNetworkClassifier()
    classifier.train(data_set)
    for feature, _ in data_set:
        print(classifier.predict(*feature))
    print([u.value for u in classifier.network.weights])
