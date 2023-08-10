# test file for the project
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from autograd.tracer import trace, Node
import pytest

import networkx as nx

# forward process test
# ignore this test using pytest
def test_forward():
    """Use a simple function to test the auto differentiation forward process"""
    def f(x): return x ** 3.
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
    