from unittest import TestCase, skip
import numpy as np

import linear_classifer
from gradient_check import check_gradient
from linear_classifer import cross_entropy_loss, softmax, softmax_with_cross_entropy


class TestGradientCheck(TestCase):

    @skip('формат входных данных для тестируемой функции подразумевает, что если Х является не вектором, а матрицей, '
          'то это batch примеров. Для тестируемой функции суммирования массивов это требование не выполняется')
    def test_on_simple_functions(self):
        self.assertTrue(check_gradient(f=lambda x: (float(x * x), 2 * x), x=np.array([3.0])))
        self.assertTrue(check_gradient(f=lambda x: (np.sum(x), np.ones_like(x)), x=np.array([3.0, 2.0])))
        self.assertTrue(check_gradient(f=lambda x: (np.sum(x), np.ones_like(x)), x=np.array([[3.0, 2.0], [1.0, 0.0]])))

    def test_on_array_1(self):
        predictions = np.array([[1, 2, -1, 1],
                                [0, 1, 1, 1],
                                [0, 2, -1, 2]], dtype=np.float)
        target_index = np.array([[2], [3], [2]])
        self.assertTrue(check_gradient(f=lambda x: softmax_with_cross_entropy(x, target_index), x=predictions))

    # def test_on_array_sum(self):
    #     def X_plus_B(X, B):
    #         result = X + B
    #         dX = np.zeros_like(X)
    #         return result, dX
    #     shape = [2, 3]
    #     test_point = np.random.randint(-5, 5, shape)*1.0
    #     B = np.random.randint(-5, 5, shape)
    #     self.assertTrue(check_gradient(f=lambda x: X_plus_B(x, B), x=test_point))



class TestLinearClassifer(TestCase):
    def test_softmax(self):
        f = softmax
        input_data = np.array([-10, 0, 10])
        sum = np.exp(input_data[0]) + np.exp(input_data[1]) + np.exp(input_data[2])
        out0 = np.exp(input_data[0]) / sum
        out1 = np.exp(input_data[1]) / sum
        out2 = np.exp(input_data[2]) / sum
        output = np.array([out0, out1, out2])
        self.assertTrue(np.all(np.isclose(f(input_data), output)))
        input_data = np.array([[-10, 0, 10]])
        self.assertTrue(np.all(np.isclose(f(input_data), output)))
        num_classes = 4
        batch_size = 3
        input_data_1 = np.random.randint(-10, 10, (num_classes,))
        input_data_2 = np.random.randint(-10, 10, (batch_size - 1, num_classes))
        input_data_2[1] = input_data_1
        output = f(input_data_2)
        self.assertTrue(np.all(np.isclose(f(input_data_1), output[1])))
        self.assertEqual(output.shape, (batch_size - 1, num_classes))

    def test_softmax_num_stability(self):
        probs = softmax(np.array([[20, 0, 0], [1000, 0, 0]]))
        assert np.all(np.isclose(probs[:, 0], 1.0))

    def test_cross_entropy_loss(self):
        f = cross_entropy_loss
        num_classes = 4
        batch_size = 5
        probs = softmax(np.random.randint(-100, 100, (batch_size, num_classes)))
        targets = np.random.randint(0, num_classes - 1, (batch_size,), dtype=np.int)
        sum = 0
        for i in range(batch_size):
            sum += f(probs[i], targets[i])
        print(sum / batch_size, f(probs, targets))
        output = f(probs, targets)
        self.assertEqual(sum / batch_size, output)
        self.assertIsInstance(output, float)

    def test_softmax_with_cross_entropy_single(self):
        f = softmax_with_cross_entropy
        num_classes = 4
        for i in range(10):
            np.random.seed(i)
            input_data = np.random.randint(-100, 100, (num_classes,)).astype(np.float)
            target_index = np.random.randint(0, num_classes - 1)
            self.assertTrue(check_gradient(lambda x: f(x, target_index), input_data))
            loss, dpredictions = f(input_data, target_index)
            self.assertEqual(dpredictions.shape, input_data.shape)
            self.assertIsInstance(loss, float)

    def test_softmax_with_cross_entropy_batch(self):
        f = softmax_with_cross_entropy
        num_classes = 4
        batch_size = 5
        for i in range(10):
            np.random.seed(i)
            input_data = np.random.randint(-100, 100, (num_classes,)).astype(np.float)
            target_index = np.random.randint(0, num_classes - 1)
            self.assertTrue(check_gradient(lambda x: f(x, target_index), input_data))

    def test_linear_softmax(self):
        batch_size = 2
        num_classes = 2
        num_features = 3
        np.random.seed(42)
        X = np.random.randint(-1, 3, size=(batch_size, num_features)).astype(np.float)
        W = np.random.randint(-1, 3, size=(num_features, num_classes)).astype(np.float)
        target_index = np.ones(batch_size, dtype=np.int)

        loss, dW = linear_classifer.linear_softmax(X, W, target_index)
        self.assertTrue(check_gradient(lambda w: linear_classifer.linear_softmax(X, w, target_index), W))
