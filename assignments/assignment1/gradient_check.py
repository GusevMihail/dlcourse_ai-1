import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape

    if len(x.shape) == 1:  # input is single sample
        print('run single-point version of gradient check')
        return check_gradient_single(f, x, delta, tol)
    else:
        print('run batch version of gradient check')
        return check_gradient_batch(f, x, delta, tol)



def check_gradient_single(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, shape is either (N) or (batch_size, N) - initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    fx, analytic_grad = f(x)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        fx, analytic_grad = f(x)
        analytic_grad_at_ix = analytic_grad[ix]
        delta_arr = np.zeros(x.shape, dtype=np.float)
        delta_arr[ix] = delta
        x1 = (x + delta_arr)
        x2 = (x - delta_arr)
        numeric_grad_at_ix = (f(x1)[0] - f(x2)[0]) / (2 * delta)
        # print('')
        # print(delta_arr)
        # print(f(x[i] + delta_arr)[0], f(x[i] - delta_arr)[0], f(x + delta_arr)[0] - f(x - delta_arr)[0])
        # print('ng', numeric_grad_at_ix)
        # print('ag', analytic_grad_at_ix)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f"
                  % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            it.close()
            return False
        # else:  # DEBUG
        #     print("Gradients at %s. Analytic: %2.5f, Numeric: %2.5f" % (
        #         ix, analytic_grad_at_ix, numeric_grad_at_ix))
        it.iternext()

    print("Gradient check passed!")
    it.close()
    return True


def check_gradient_batch(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, shape is (batch_size, N) - initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    fx, analytic_grad = f(x)
    batch_size = x.shape[0]
    numeric_grad = np.zeros_like(analytic_grad, dtype=np.float)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        fx, analytic_grad = f(x)

        xcopy = x.copy()
        xcopy[ix] += delta
        # print('x1\n', xcopy)
        fx1 = f(xcopy)[0]
        xcopy[ix] -= delta * 2
        # print('x2\n', xcopy)
        fx2 = f(xcopy)[0]
        numeric_grad[ix] = (fx1 - fx2) * batch_size / (2 * delta)  # домножение на размер батча - эмпирическое решение,
        # нет гарантий, что оно верно в общем случае.
        it.iternext()
    it.close()

    # DEBUG
    print('analytic_grad\n', analytic_grad, '\n')
    print('numeric_grad\n', numeric_grad, '\n')

    if np.all(np.isclose(numeric_grad, analytic_grad)):
        print("Gradient check passed!")
        return True
    else:
        print("Gradients are different!")
        return False
