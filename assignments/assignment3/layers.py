import numpy as np

from assignment2.layers import l2_regularization, softmax_with_cross_entropy, ReLULayer, FullyConnectedLayer


# def l2_regularization(W, reg_strength):
#     '''
#     Computes L2 regularization loss on weights and its gradient
#
#     Arguments:
#       W, np array - weights
#       reg_strength - float value
#
#     Returns:
#       loss, single value - l2 regularization loss
#       gradient, np.array same shape as W - gradient of weight by l2 loss
#     '''
#     # TODO: Copy from previous assignment
#     raise Exception("Not implemented!")
#
#     return loss, grad
#
#
# def softmax_with_cross_entropy(predictions, target_index):
#     '''
#     Computes softmax and cross-entropy loss for model predictions,
#     including the gradient
#
#     Arguments:
#       predictions, np array, shape is either (N) or (batch_size, N) -
#         classifier output
#       target_index: np array of int, shape is (1) or (batch_size) -
#         index of the true class for given sample(s)
#
#     Returns:
#       loss, single value - cross-entropy loss
#       dprediction, np array same shape as predictions - gradient of predictions by loss value
#     '''
#     # TODO copy from the previous assignment
#     raise Exception("Not implemented!")
#     return loss, dprediction
#


# class ReLULayer:
#     def __init__(self):
#         pass
#
#     def forward(self, X):
#         # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
#
#     def backward(self, d_out):
#         # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
#         return d_result
#
#     def params(self):
#         return {}


# class FullyConnectedLayer:
#     def __init__(self, n_input, n_output):
#         self.W = Param(0.001 * np.random.randn(n_input, n_output))
#         self.B = Param(0.001 * np.random.randn(1, n_output))
#         self.X = None
#
#     def forward(self, X):
#         # TODO copy from the previous assignment
#         raise Exception("Not implemented!")
#
#     def backward(self, d_out):
#         # TODO copy from the previous assignment
#
#         raise Exception("Not implemented!")
#         return d_input
#
#     def params(self):
#         return { 'W': self.W, 'B': self.B }

class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        out = np.array([out_width, out_height])

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one X/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        shift_1 = self.filter_size - self.filter_size // 2 - 1
        shift_2 = self.filter_size // 2 + 1
        for y in range(out_height):
            for x in range(out_width):
                X1 = X[:, x - shift_1:x + shift_2, y - shift_1:y + shift_2, :]
                out[x, y] = np.sum(
                        np.dot(X1.reshape([batch_size, self.filter_size ** 2 * channels]),
                                   self.W.value.reshape([self.filter_size ** 2 * self.in_channels, self.out_channels])
                                   ) + self.B.value)

        return out.reshape([batch_size, self.filter_size, self.filter_size, self.out_channels])

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass

        raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output X/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
