# test file for the project
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad



# test for correct differentiation on simple function
def test_grad():
    """Use common functions to test grad function"""
    # Common function families include: polynomials, logarithms, exponentials, trigonometric, hyperbolic
    x = 2.5
    assert grad(lambda x: x ** 3.)(x) == 3 * x ** 2., "x^3 test failed!"
    assert grad(lambda x: np.log(x))(x) == 1 / x, "log test failed!"
    assert grad(lambda x: np.exp(x))(x) == np.exp(x), "exp test failed!"
    assert grad(lambda x: np.sin(x))(x) == np.cos(x), "sin test failed!"
    assert grad(lambda x: np.cos(x))(x) == -np.sin(x), "cos test failed!"
    assert grad(lambda x: np.tan(x))(x) == 1 / np.cos(x) ** 2., "tan test failed!"
    x = 0.5
    assert grad(lambda x: np.arcsin(x))(x) == 1 / np.sqrt(1 - x ** 2.), "arcsin test failed!"
    assert grad(lambda x: np.arccos(x))(x) == -1 / np.sqrt(1 - x ** 2.), "arccos test failed!"
    assert grad(lambda x: np.arctan(x))(x) == 1 / (1 + x ** 2.), "arctan test failed!"
    x = 1.5
    assert grad(lambda x: np.sinh(x))(x) == np.cosh(x), "sinh test failed!"
    assert grad(lambda x: np.cosh(x))(x) == np.sinh(x), "cosh test failed!"
    assert grad(lambda x: np.tanh(x))(x) == 1 / np.cosh(x) ** 2., "tanh test failed!"

# test higher order derivatives
def test_higher_order():
    """Use common functions to test grad function"""
    # for 3, 4, 5 order derivatives, we only test log
    v = 2.5
    assert grad(grad(grad(lambda x: np.log(x))))(v) == 2 / v ** 3., "3rd order log test failed!"
    assert grad(grad(grad(grad(lambda x: np.log(x)))))(v) == -6 / v ** 4., "4th order log test failed!"
    assert abs(grad(grad(grad(grad(grad(lambda x: np.log(x))))))(v) - 24. / v ** 5.) < 1e-8, "5th order log test failed!"

