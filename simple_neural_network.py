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
        # Will be a tuple after first forward function
        self.units = None

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
        utop = self._forward(*units)
        return utop

    @abstractmethod
    def _backward(self):
        raise NotImplementedError

    def backward(self):
        """
        Run the backward process to update the gradient of each unit in the gate

        Note: the method should not be overrode! Override _backward method instead.
        """
        # self.units = self.units_history.pop()
        # self.utop = self.utop_history.pop()
        self._backward()

    def set_utop_gradient(self, gradient):
        """
        Must be called before backward function for the final <Gate> instance
        todo: how do we check if this method is called as the gradient default is 0
        """
        self.utop.gradient = gradient


class AddGate(Gate):
    def __init__(self):
        super(AddGate, self).__init__()

    def _forward(self, u0, u1):
        self.utop = Unit(u0.value + u1.value)
        return self.utop

    def _backward(self):
        self.units[0].gradient += 1 * self.utop.gradient
        self.units[1].gradient += 1 * self.utop.gradient


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
        # Store the <Gate> instance created in forward method here, append as a tuple
        # Note the sequence is the backward sequence
        self._gates_history = []
        # Store the utop corresponding to the gates above
        self._utop_history = []

    @abstractmethod
    def _forward(self, *units):
        raise NotImplementedError

    def _backward(self):
        assert self._gates_history, "Must append the <Gate> instances created in forward to self._gates_history"
        assert self._utop_history, "Must append the utop created in forward to self._utop_history"
        gates = self._gates_history.pop()
        for g in gates:
            g.backward()
        # Reset the utop to corresponding gates if the utop_history has >=2 items
        # We set the utop to the last second one in history, because the last one is just current utop
        self._utop_history.pop()
        if self._utop_history:
            self.utop = self._utop_history[-1]

    def pull_weights(self, learning_rate):
        """
        Adjust all the weights according to the gradients
        Should be called after forward and backward process
        """
        for w in self.weights:
            w.value += learning_rate * w.gradient
        # Reset all the weights' gradient to 0
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

    def _forward(self, x, y):
        multi_gate0 = MultiplyGate()
        multi_gate1 = MultiplyGate()
        add_gate0 = AddGate()
        add_gate1 = AddGate()
        multi_gate0.forward(self.a, x)
        multi_gate1.forward(self.b, y)
        add_gate0.forward(multi_gate0, multi_gate1)
        self.utop = add_gate1.forward(add_gate0, self.c)
        # The sequence is the backward sequence
        self._gates_history.append((add_gate1, add_gate0, multi_gate1, multi_gate0))
        self._utop_history.append(self.utop)
        return self.utop

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

    def _forward(self, x, y):
        relu_gate = ReLUGate()
        self.linear_network.forward(x, y)
        self.utop = relu_gate.forward(self.linear_network)
        self._gates_history.append((relu_gate, self.linear_network))
        self._utop_history.append(self.utop)
        return self.utop

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

    def _forward(self, x, y):
        self.neuro0.forward(x, y)
        self.neuro1.forward(x, y)
        self.utop = self.linear_network.forward(self.neuro0, self.neuro1)
        self._gates_history.append((self.linear_network, self.neuro0, self.neuro1))
        self._utop_history.append(self.utop)
        return self.utop

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

    def _forward(self):
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

    def _backward(self):
        # TODO: reset all the weights' gradient to 0, should also reset all other units' gradient?
        self.add_gate_final.backward()
        self.multi_gate_alpha.backward()
        for gate in reversed(self.add_gates_sigma_r):
            gate.backward()
        for gate in reversed(self.multi_gates_r):
            gate.backward()
        for gate in reversed(self.add_gates_sigma_l):
            gate.backward()
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
            self._loss_by_step.append(loss_network.value)
            loss_network.set_utop_gradient(-1)
            loss_network.backward()
            loss_network.pull_weights(learning_rate)

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self._loss_by_step)
        plt.ylabel('loss')
        plt.show()

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
    classifier = NeuralNetworkClassifier()
    # classifier.simple_train(data_set)
    classifier.train(data_set, learning_rate=0.01, steps=200)
    classifier.plot_loss()
    print('---')
    import matplotlib.pyplot as plt

    for x in range(-30, 30):
        for y in range(-30, 30):
            _x = x * 0.1
            _y = y * 0.1
            label = classifier.predict(_x, _y)
            color = '#a1d5ed' if label > 0 else '#efaabd'
            plt.plot(_x, _y, color, marker='*')

    for feature, label in data_set:
        print(classifier.predict(*feature))
        color = 'b' if label > 0 else 'r'
        plt.plot(*feature, color + 'o')
    print([u.value for u in classifier.network.weights])

    plt.show()
