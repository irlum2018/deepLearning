import numpy as np

def check_gradient(f, x, delta=1e-5, tol = 1e-4):
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
    
    fx, analytic_grad = f(x)
    analytic_grad = analytic_grad.copy()

    assert analytic_grad.shape == x.shape
    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # ix tuple iterating through cells of nd.array
        ix = it.multi_index        
        analytic_grad_at_ix = analytic_grad[ix]
		# compute value of numeric gradient of f to idx
        x1 = x.copy()
        x1[ix] += delta
        x2 = x.copy()
        x2[ix] -= delta
        numeric_grad_pos, _ = f(x1)
        numeric_grad_neg, _ = f(x2)
        
        # store this value to compare with analytical formula
        numeric_grad_at_ix =  (numeric_grad_pos - numeric_grad_neg) / (2 * delta)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
