import numpy as np

from assignment3.layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv1 = ConvolutionalLayer(in_channels=3, out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.mp1 = MaxPoolingLayer(pool_size=4, stride=4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.mp2 = MaxPoolingLayer(pool_size=4, stride=4)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(n_input=2 * 2 * conv2_channels, n_output=n_output_classes)

    def clear_gradients(self):
        for param in self.params().values():
            param.grad[:] = 0.0

    def forward(self, X):
        out1 = self.conv1.forward(X)
        out2 = self.relu1.forward(out1)
        out3 = self.mp1.forward(out2)
        out4 = self.conv2.forward(out3)
        out5 = self.relu2.forward(out4)
        out6 = self.mp2.forward(out5)
        out7 = self.flat.forward(out6)
        out8 = self.fc.forward(out7)
        return out8

    def backward(self, d_out):
        d7 = self.fc.backward(d_out)
        d6 = self.flat.backward(d7)
        d5 = self.mp2.backward(d6)
        d4 = self.relu2.backward(d5)
        d3 = self.conv2.backward(d4)
        d2 = self.mp1.backward(d3)
        d1 = self.relu1.backward(d2)
        dX = self.conv1.backward(d1)
        return dX

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        self.clear_gradients()
        preds = self.forward(X)
        loss, d_preds = softmax_with_cross_entropy(preds, y)
        self.backward(d_preds)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)

        preds = self.forward(X)
        y_pred = np.argmax(preds, axis=1)
        return y_pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        def rename_keys(input_dict: dict, key_prefix: str) -> dict:
            result_dict = {}
            for key, value in input_dict.items():
                result_dict[key_prefix + key] = value
            return result_dict

        result.update(rename_keys(self.conv1.params(), 'conv1_'))
        result.update(rename_keys(self.conv2.params(), 'conv2_'))
        result.update(rename_keys(self.fc.params(), 'fc_'))

        return result


class TestNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        # self.conv1 = ConvolutionalLayer(in_channels=3, out_channels=conv1_channels, filter_size=3, padding=1)
        # self.relu1 = ReLULayer()
        # self.mp1 = MaxPoolingLayer(pool_size=4, stride=4)
        # self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.conv2 = ConvolutionalLayer(3, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.mp2 = MaxPoolingLayer(pool_size=4, stride=4)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(n_input=8 ** 2 * conv2_channels, n_output=n_output_classes)

    def clear_gradients(self):
        for param in self.params().values():
            param.grad[:] = 0.0

    def forward(self, X):
        out3 = X
        # out1 = self.conv1.forward(X)
        # out2 = self.relu1.forward(out1)
        # out3 = self.mp1.forward(out2)
        out4 = self.conv2.forward(out3)
        out5 = self.relu2.forward(out4)
        out6 = self.mp2.forward(out5)
        out7 = self.flat.forward(out6)
        out8 = self.fc.forward(out7)
        return out8

    def backward(self, d_out):
        d7 = self.fc.backward(d_out)
        d6 = self.flat.backward(d7)
        d5 = self.mp2.backward(d6)
        d4 = self.relu2.backward(d5)
        d3 = self.conv2.backward(d4)
        # d2 = self.mp1.backward(d3)
        # d1 = self.relu1.backward(d2)
        # dX = self.conv1.backward(d1)
        return d3

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        self.clear_gradients()
        preds = self.forward(X)
        loss, d_preds = softmax_with_cross_entropy(preds, y)
        self.backward(d_preds)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)

        preds = self.forward(X)
        y_pred = np.argmax(preds, axis=1)
        return y_pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        def rename_keys(input_dict: dict, key_prefix: str) -> dict:
            result_dict = {}
            for key, value in input_dict.items():
                result_dict[key_prefix + key] = value
            return result_dict

        # result.update(rename_keys(self.conv1.params(), 'conv1_'))
        result.update(rename_keys(self.conv2.params(), 'conv2_'))
        result.update(rename_keys(self.fc.params(), 'fc_'))

        return result
