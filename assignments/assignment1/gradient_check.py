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
        return check_gradient_single(f, x, delta, tol)
        # analytic_grad = analytic_grad.copy()
        # numeric_grad = np.zeros(x.shape, dtype=np.float)

        # We will go through every dimension of x and compute numeric
        # derivative for it
        # if len(x.shape) == 1:  # input is single sample
        #     x = x[np.newaxis, :]

        # for i in range(x.shape[0]):


        # print("Gradient check passed!")
        # return True


def check_gradient_single(f, x, delta=1e-5, tol=1e-4):

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
        # print('raw ng', numeric_grad_at_ix)
        # print(delta_arr)
        # print(f(x[i] + delta_arr)[0], f(x[i] - delta_arr)[0], f(x + delta_arr)[0] - f(x - delta_arr)[0])
        # print('ng', numeric_grad_at_ix)
        # print('ag', analytic_grad_at_ix)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f"
                  % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            it.close()
            return False
        else:  # DEBUG
            print("Gradients at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                ix, analytic_grad_at_ix, numeric_grad_at_ix))
        it.iternext()

    print("Gradient check passed!")
    it.close()

    return True
