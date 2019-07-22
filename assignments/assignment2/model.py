import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc_layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer_1 = ReLULayer()
        self.fc_layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def clear_gradients(self):
        for param in self.params().values():
            param.grad[:] = 0.0

    def forward(self, X):
        out1 = self.fc_layer_1.forward(X)
        out2 = self.relu_layer_1.forward(out1)
        preds = self.fc_layer_2.forward(out2)
        return preds

    def backward(self, d_preds):
        d_fc2 = self.fc_layer_2.backward(d_preds)
        d_relu1 = self.relu_layer_1.backward(d_fc2)
        d_fc1 = self.fc_layer_1.backward(d_relu1)
        return d_fc1

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.clear_gradients()

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        preds = self.forward(X)
        softmax_loss, d_preds = softmax_with_cross_entropy(preds, y)
        d_X = self.backward(d_preds)

        loss = softmax_loss  # temp code
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        #         # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        # result = list(self.fc_layer_1.params().values()) + \
        #          list(self.fc_layer_2.params().values()) + \
        #          list(self.relu_layer_1.params().values())
        result = {}

        def rename_keys(input_dict: dict, key_prefix: str) -> dict:
            result_dict = {}
            for key, value in input_dict.items():
                result_dict[key_prefix + key] = value
            return result_dict

        result.update(rename_keys(self.fc_layer_1.params(), 'fc1_'))
        result.update(rename_keys(self.fc_layer_2.params(), 'fc2_'))
        result.update(rename_keys(self.relu_layer_1.params(), 'relu1_'))

        return result
