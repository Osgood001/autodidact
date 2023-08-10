# test file for the project
import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
import numpy as onp
from autograd.differential_operators import grad
from autograd.tracer import trace, Node, primitive, Box, new_box
from autograd.core import backward_pass, make_vjp, primitive_vjps
import pytest

# other packages
import networkx as nx
import matplotlib.pyplot as plt

#### Unit Level test ####

# test the vjp completeness
@pytest.mark.skip(reason="Not sure what primitive_vjps is")
def test_vjp_completeness():
    """Ensure that all functions in numpy_wrapper has a vjp"""
    # define a set of numpy functions that we want to test
    numpy_functions = [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh, np.exp, np.log]
    # test the vjp completeness
    for f in numpy_functions:
        # first we trace the function
        x = 2.5
        start_node = Node.new_root()
        end_value, end_node = trace(start_node, f, x)
        f = end_node.recipe[0]
        # assert primitive_vjps[f] , "VJP completeness test failed!"

# test the numpy wrapper
# @pytest.mark.skip(reason="Entangled with other tests")
def test_numpy_wrapper():
    """See if we have a full set of numpy functions that can be traced (by randomly sampling from the set)"""
    # define a set of numpy functions that we want to test
    numpy_functions = [np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh, np.exp, np.log]
    # randomly sample from the set and test
    for i in range(10):
        # randomly sample a function
        g = onp.random.choice(numpy_functions)
        # randomly sample an input
        x = onp.random.rand()
        # test the function
        assert g(x) == g(ArrayBox(x, 0, None)), "Numpy wrapper test failed!"
    
#### Component Level test####

# define a global function f(x) = x^3 for the following tests
def f(x): return x ** 3. # DON'T CHANGE THIS LINE, since we explicitly write the derivative of x^3 below

# test the VJP that takes a function and a value, but return a function that takes a vector
def test_make_vjp():
    """Test the make_vjp function"""
    x = 2.5
    vjp, end_value = make_vjp(f, x)
    assert vjp(1.) == 3 * x ** 2., "make_vjp test failed!"
    assert end_value == f(x), "make_vjp test failed!" 

# test the primitive function wrapper
def test_primitive():
    """Primitve wrapper should receive a function and return a wrapped function that can be traced"""
    # wrap the function
    f_wrapped = primitive(f)
    # test the wrapped function
    x = 2.5
    assert f_wrapped(x) == f(x), "Primitive wrapper test failed!"
    # box class should be able to call the wrapped function
    x_box = new_box(x, 0, None)
    assert f_wrapped(x_box) == f(x), "Primitive wrapper test failed!"
    
    

# test the Node, Box class that are used to wrap values and functions
def test_node():
    """See if we can wrap a value in a Node"""
    x = 2.5
    node = Node(x, lambda x: x, (x,), {}, (0,), ())
    assert node.recipe[1] == x, "Node test failed!"
    assert node.recipe[2][0] == x, "Node test failed!"
    assert node.recipe[3] == {}, "Node test failed!"
    assert node.recipe[4] == (0,), "Node test failed!"
    # also wrap value in a box
    x_box = Box(x, 0, None)
    assert x_box._value == x, "Box test failed!"
    assert x_box._trace_id == 0, "Box test failed!"
    # wrap an ndarray in a box
    x_ndarray = onp.array([1., 2., 3.])
    x_box = ArrayBox(x_ndarray, 0, None)
    # see if we can do basic operations on the box
    assert x_box[0] == 1., "Box test failed!"
    assert (x_box + 1. == onp.array([2., 3., 4.])).all(), "Box test failed!"
    assert (x_box * 2. == onp.array([2., 4., 6.])).all(), "Box test failed!"
    assert (x_box ** 2. == onp.array([1., 4., 9.])).all(), "Box test failed!"
    assert (x_box / 2. == onp.array([0.5, 1., 1.5])).all(), "Box test failed!"
    assert (x_box % 2. == onp.array([1., 0., 1.])).all(), "Box test failed!"
    


   

##### Function level tests #####

# reverse process test
def test_reverse():
    """Test the backward_pass funciton that traverse computation grap and compute gradients"""
    # construct a simple computational graph
    x = 2.5
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, f, x)
    # backward pass
    g = 1.
    outgrad = backward_pass(g, end_node)
    assert outgrad == 3 * x ** 2., "reverse test failed!"

    # construct a more complicated computational graph using a complex function that are composed of multiple primitive functions
    def g(x): return np.sin(np.log(np.tanh(4 * x ** 2.)))
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, g, x)
    # backward pass
    g = 1.
    outgrad = backward_pass(g, end_node)
    assert abs(outgrad - 1. / np.cosh(4 * x ** 2.) ** 2. * 8 * x * np.cos(np.log(np.tanh(4 * x ** 2.))) / np.tanh(4 * x ** 2.)) < 1e-8, "reverse test failed!"
    

# forward process test
def test_forward():
    """Use a simple function to test the auto differentiation forward process"""
    x = 2.5
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, f, x)
    assert end_value == f(x), "forward test failed!" 
    assert end_node.recipe[1] == f(x), "forward test failed!"
    assert end_node.recipe[2][0] == x, "forward test failed!"
    assert end_node.recipe[3] == {}, "forward test failed!"
    assert end_node.recipe[4] == (0,), "forward test failed!"

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

# test for correct differentiation on 2D function
def test_2D():
    """Use a simple 2D function to test grad function"""
    def f(x, y): return x ** 2. + 2 * y ** 2.
    x = 2.5
    y = 3.5
    assert grad(f, 0)(x, y) == 2 * x, "2D test failed!"
    assert grad(f, 1)(x, y) == 4 * y, "2D test failed!"

# a helper function that plot the computational graph given an end node
def plot_computational_graph(end_node):
    """Plot the computational graph of a 2D function"""
    # build the computational graph
    nodes = [end_node]
    node_id = {end_node: 0}
    for node in nodes:
        for parent in node.parents:
            if parent not in node_id:
                node_id[parent] = len(nodes)
                nodes.append(parent)
    # plot the computational graph
    # define a figure
    fig = plt.figure(figsize=(10, 10))
    # define a plot
    ax = fig.add_subplot(1, 1, 1)
    # visualize the computational graph using networkx
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node_id[node], label=str(node))
        for parent in node.parents:
            G.add_edge(node_id[parent], node_id[node])
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='w')
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
    nx.draw_networkx_labels(G, pos, ax=ax)
    plt.show()
    # save the figure
    fig.savefig('computational_graph.png')
   

# plot the computational graph of a simple function
@pytest.mark.skip(reason="Plot is heuristic, not a test")
def plot_test():
    """Plot the computational graph of a 2D function"""
    def f(x, y): return x * y + np.sin(x) 
    # build the computational graph
    x = 2.
    y = 3.
    g = lambda x: f(x, y)
    start_node = Node.new_root()
    end_value, end_node = trace(start_node, g, x)
    # plot the computational graph
    plot_computational_graph(end_node)
    

if __name__ == '__main__':
    def f(x): return np.tanh(x) 
    x = np.linspace(-7, 7, 200)
    plt.plot(x, f(x))
    plt.plot(x, grad(f)(x))
    plt.plot(x, grad(grad(f))(x))
    plt.plot(x, grad(grad(grad(f)))(x))
    plt.plot(x, grad(grad(grad(grad(f))))(x))
    
    plt.show()
    plt.savefig('test.png')