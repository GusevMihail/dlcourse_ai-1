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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        """
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        """

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
        batch_size, height, width, channels = X.shape
        p = self.padding
        self.X = np.zeros([batch_size, height + 2 * p, width + 2 * p, channels])
        self.X[:, p:p + height, p:p + width, :] = X
        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        out = np.zeros([batch_size, out_width, out_height, self.out_channels])

        # TO DO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one X/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        shift_1 = self.filter_size - self.filter_size // 2 - 1
        shift_2 = self.filter_size // 2 + 1
        for y in range(out_height):
            for x in range(out_width):
                Xc = self.X[:, x:x + self.filter_size, y:y + self.filter_size, :]  # X crop
                Xcr = Xc.reshape([batch_size, self.filter_size ** 2 * self.in_channels])  # X crop reshaped
                Wr = self.W.value.reshape([self.filter_size ** 2 * self.in_channels, self.out_channels])  # W reshaped
                XWB = np.dot(Xcr, Wr) + self.B.value
                out[:, x, y, :] = XWB
        # print(self.X[0, :, :, 0])
        return out

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

        d_input = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                Wr = self.W.value.reshape([self.filter_size ** 2 * self.in_channels, self.out_channels])  # W reshaped
                Wrt = np.transpose(Wr)  # W reshaped transposed
                d_input_c = np.dot(d_out[:, x, y, :], Wrt)  # dX cropped
                d_input_cr = d_input_c.reshape(
                    [batch_size, self.filter_size, self.filter_size, self.in_channels])  # dX cropped reshaped
                d_input[:, x:x + self.filter_size, y:y + self.filter_size, :] += d_input_cr  # dX

                Xc = self.X[:, x:x + self.filter_size, y:y + self.filter_size, :]  # X crop
                Xcr = Xc.reshape([batch_size, self.filter_size ** 2 * self.in_channels])  # X crop reshaped
                Xcrt = np.transpose(Xcr)  # X cropped reshaped transposed
                dW = np.dot(Xcrt, d_out[:, x, y, :])  # dW for one sample of d_input
                dWr = dW.reshape(
                    [self.filter_size, self.filter_size, self.in_channels, self.out_channels])  # dW cropped reshaped
                self.W.grad += dWr / batch_size  # accumulate dW.
                # !Деление на BatchSize - эмпирическое решение, нет гарантий, что оно верно!
                d_out_r = d_out.reshape([batch_size * out_height * out_width, out_channels])
                self.B.grad += d_out_r.mean(axis=0).reshape(self.B.value.shape)
                # self.B.grad += d_out_r.sum(axis=0).reshape(self.B.value.shape) / batch_size
                # !Деление на BatchSize - эмпирическое решение, нет гарантий, что оно верно!
        # print(d_input[0, :, :, 0])

        if self.padding == 0:
            return d_input
        else:
            p = self.padding
            return d_input[:, p:-p, p:-p, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        assert self.stride >= self.pool_size, 'stride must be >= pool_size'

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        out = np.zeros([batch_size, out_width, out_height, channels], dtype=np.float)
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output X/y dimension
        for y in range(out_height):
            for x in range(out_width):
                ps = self.pool_size
                st = self.stride
                Xc = self.X[:, x * st: x * st + ps, y * st: y * st + ps, :]  # X cropped
                out[:, x, y, :] = np.amax(np.amax(Xc, axis=1), axis=1)
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                for b in range(batch_size):
                    for c in range(channels):
                        ps = self.pool_size
                        st = self.stride
                        Xc = self.X[b, x * st: x * st + ps, y * st: y * st + ps, c]  # X cropped
                        # ind_max = np.where(Xc == Xc.max())
                        ind_max = np.unravel_index(np.argmax(Xc), Xc.shape)
                        # dX[b, x * st + ind_max[0][0], y * st + ind_max[1][0], c] = d_out[b, x, y, c]
                        dX[b, x * st + ind_max[0], y * st + ind_max[1], c] = d_out[b, x, y, c]
        return dX

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None
        self.batch_size = None
        self.height = None
        self.width = None
        self.channels = None

    def forward(self, X):
        self.batch_size, self.height, self.width, self.channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, height*width*channels]
        return X.reshape([self.batch_size, self.height * self.width * self.channels])

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape([self.batch_size, self.height, self.width, self.channels])

    def params(self):
        # No params!
        return {}
