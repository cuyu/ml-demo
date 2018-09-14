#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implement a simple neural network, using modular.
Support input features only have two dimensions
"""
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
    def backward(self, overlay=False):
        """
        Run the backward process to update the gradient of each unit in the gate
        :param overlay: set to True to overlay previous gradient, otherwise will reset and recalculate the gradient
        """
        raise NotImplementedError

    def set_utop_gradient(self, gradient):
        """
        Must be called before backward function for the final <Gate> instance
        todo: how do we check if this method is called as the gradient default is 0
        """
        self.utop.gradient = gradient


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

    def backward(self, overlay=False):
        if not overlay:
            self.u0.gradient = 0
            self.u1.gradient = 0
        self.u0.gradient += 1 * self.utop.gradient
        self.u1.gradient += 1 * self.utop.gradient


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

    def backward(self, overlay=False):
        """
        Use the chain rule, assume f as the final output:
        d(f)/d(u0) = d(f)/d(utop) * d(utop)/d(u0)
                   = utop.gradient * u1.value
        """
        if not overlay:
            self.u0.gradient = 0
            self.u1.gradient = 0
        self.u0.gradient += self.u1.value * self.utop.gradient
        self.u1.gradient += self.u0.value * self.utop.gradient


class SigmoidGate(Gate):
    def __init__(self):
        super(SigmoidGate, self).__init__()
        self.u0 = None

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(1 / (1 + math.exp(-self.u0.value)))
        return self.utop

    def backward(self, overlay=False):
        if not overlay:
            self.u0.gradient = 0
        self.u0.gradient += self.u0.value * (1 - self.u0.value) * self.utop.gradient


class ReLUGate(Gate):
    def __init__(self):
        super(ReLUGate, self).__init__()
        self.u0 = None

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(max(0, self.u0.value))
        return self.utop

    def backward(self, overlay=False):
        """
        Here, we define the derivative at x=0 to 0
        Refer to https://www.quora.com/How-do-we-compute-the-gradient-of-a-ReLU-for-backpropagation
        """
        if not overlay:
            self.u0.gradient = 0

        if self.u0.value > 0:
            self.u0.gradient += 1 * self.utop.gradient
        else:
            self.u0.gradient += 0 * self.utop.gradient


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
    def backward(self, overlay=False):
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

    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.a = Unit(1.0)
        self.b = Unit(1.0)
        self.c = Unit(1.0)
        self.multi_gate0 = MultiplyGate()
        self.multi_gate1 = MultiplyGate()
        self.add_gate0 = AddGate()
        self.add_gate1 = AddGate()

    def forward(self, x, y):
        self.multi_gate0.forward(self.a, x)
        self.multi_gate1.forward(self.b, y)
        self.add_gate0.forward(self.multi_gate0, self.multi_gate1)
        self.utop = self.add_gate1.forward(self.add_gate0, self.c)
        return self.utop

    def backward(self, overlay=False):
        self.add_gate1.backward(overlay)
        self.add_gate0.backward(overlay)
        self.multi_gate1.backward(overlay)
        self.multi_gate0.backward(overlay)

    @property
    def weights(self):
        return [self.a, self.b, self.c]

    @property
    def weights_without_bias(self):
        return [self.a, self.b]


class SingleNeuralNetwork(Network):
    """
    SingleNeuralNetwork, a.k.a. a Neuro of NeuralNetwork
    The formula is:
        f(x, y) = max(0, a * x + b * y + c)
    We can just think as it put the output of a <LinearNetwork> into a <ReLU> Gate
    """

    def __init__(self):
        super(SingleNeuralNetwork, self).__init__()
        self.linear_network = LinearNetwork()
        self.relu_gate = ReLUGate()

    def forward(self, x, y):
        self.linear_network.forward(x, y)
        self.utop = self.relu_gate.forward(self.linear_network)
        return self.utop

    def backward(self, overlay=False):
        self.relu_gate.backward(overlay)
        self.linear_network.backward(overlay)

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

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.neuro0 = SingleNeuralNetwork()
        self.neuro1 = SingleNeuralNetwork()
        self.linear_network = LinearNetwork()

    def forward(self, x, y):
        self.neuro0.forward(x, y)
        self.neuro1.forward(x, y)
        self.utop = self.linear_network.forward(self.neuro0, self.neuro1)
        return self.utop

    def backward(self, overlay=False):
        self.linear_network.backward(overlay)
        self.neuro0.backward(overlay)
        self.neuro1.backward(overlay)

    @property
    def weights(self):
        return self.neuro0.weights + self.neuro1.weights + self.linear_network.weights

    @property
    def weights_without_bias(self):
        return self.neuro0.weights_without_bias + self.neuro1.weights_without_bias + self.linear_network.weights_without_bias


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
        self.add_gates_sigma_l = [AddGate() for _ in range(feature_number - 1)]
        weight_number = len(self.network.weights_without_bias)
        # Multiply in `w_i*w_i`
        self.multi_gates_r = [MultiplyGate() for _ in range(weight_number)]
        # Add of ∑ in `∑w_i*w_i`
        self.add_gates_sigma_r = [AddGate() for _ in range(weight_number - 1)]
        # Multiply alpha in `α*∑w_i*w_i`
        self.multi_gate_alpha = MultiplyGate()
        # The final add
        self.add_gate_final = AddGate()

    def forward(self):
        # Calculate ∑max(0,−y_i*f(X_i)+1)
        for i in range(len(self.relu_gates_l)):
            feature = self.data_set[i][0]
            label = self.data_set[i][1]
            self.network.forward(*[Unit(x) for x in feature])
            self.multi_gates_l[i].forward(Unit(-label), self.network)
            self.add_gates_l[i].forward(self.multi_gates_l[i], Unit(1.0))
            self.relu_gates_l[i].forward(self.add_gates_l[i])
        self.add_gates_sigma_l[0].forward(self.relu_gates_l[0], self.relu_gates_l[1])
        for i in range(1, len(self.add_gates_sigma_l)):
            self.add_gates_sigma_l[i].forward(self.add_gates_sigma_l[i - 1], self.relu_gates_l[i + 1])
        # Calculate α*∑w_i*w_i
        for i in range(len(self.multi_gates_r)):
            self.multi_gates_r[i].forward(self.network.weights_without_bias[i], self.network.weights_without_bias[i])
        self.add_gates_sigma_r[0].forward(self.multi_gates_r[0], self.multi_gates_r[1])
        for i in range(1, len(self.add_gates_sigma_r)):
            self.add_gates_sigma_r[i].forward(self.add_gates_sigma_r[i - 1], self.multi_gates_r[i + 1])
        self.multi_gate_alpha.forward(Unit(self.alpha), self.add_gates_sigma_r[-1])
        self.utop = self.add_gate_final.forward(self.add_gates_sigma_l[-1], self.multi_gate_alpha)

    def backward(self):
        self.add_gate_final.backward()
        self.multi_gate_alpha.backward()
        for gate in reversed(self.add_gates_sigma_r):
            gate.backward()
        for gate in reversed(self.multi_gates_r):
            gate.backward()
        for gate in reversed(self.add_gates_sigma_l):
            gate.backward()
        overlay = False
        for i in reversed(range(len(self.relu_gates_l))):
            self.relu_gates_l[i].backward()
            self.add_gates_l[i].backward()
            self.multi_gates_l[i].backward()
            self.network.backward(overlay=overlay)
            overlay = True

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
                self.network.gradient = pull
                self.network.backward()
                self.network.pull_weights(learning_rate)

    def train(self, data_set, learning_rate=0.01, steps=100):
        """
        Train the classifier using loss function (with all the features and labels)
        """
        loss_network = LossNetwork(self.network, data_set)
        for _ in range(steps):
            loss_network.forward()
            print(loss_network.value)
            loss_network.set_utop_gradient(-1)
            loss_network.backward()
            loss_network.pull_weights(learning_rate)

    def predict(self, x, y):
        return self.network.forward(Unit(x), Unit(y)).value


class LinearClassifier(BasicClassifier):
    def __init__(self):
        network = LinearNetwork()
        super(LinearClassifier, self).__init__(network)


class NeuralNetworkClassifier(BasicClassifier):
    def __init__(self):
        network = NeuralNetwork()
        super(NeuralNetworkClassifier, self).__init__(network)


if __name__ == '__main__':
    data_set = [
        ([1.2, 0.7], 1),
        ([-0.3, -0.5], -1),
        ([3.0, 0.1], 1),
        ([-0.1, -1.0], -1),
        ([-1.0, 1.1], -1),
        ([2.1, -3.0], 1),
    ]
    classifier = LinearClassifier()
    # classifier.simple_train(data_set)
    classifier.train(data_set, steps=200)
    print('---')
    for feature, _ in data_set:
        print(classifier.predict(*feature))
    print([u.value for u in classifier.network.weights])
