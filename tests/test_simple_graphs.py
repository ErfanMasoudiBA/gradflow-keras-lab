from gradflow.gates import square_forward, square_backward
from gradflow.simple_graphs import forward_xyz_graph, backward_xyz_graph
from gradflow.grad_check import numerical_gradient_3vars


def test_square_forward():
    assert square_forward(5) == 25
    assert square_forward(-3) == 9


def test_square_backward():
    dx = square_backward(x=5, upstream_grad=2)

    assert dx == 20


def test_forward_xyz_graph():
    L, cache = forward_xyz_graph(x=2, y=3, z=4)

    assert cache["a"] == 6
    assert cache["b"] == 10
    assert L == 100


def test_backward_xyz_graph():
    L, cache = forward_xyz_graph(x=2, y=3, z=4)
    grads = backward_xyz_graph(cache)

    assert L == 100
    assert grads["x"] == 60
    assert grads["y"] == 40
    assert grads["z"] == 20

def test_backward_matches_numerical_gradient():
    def fn(x, y, z):
        L, _ = forward_xyz_graph(x, y, z)
        return L

    x = 2.0
    y = 3.0
    z = 4.0

    L, cache = forward_xyz_graph(x, y, z)
    analytical_grads = backward_xyz_graph(cache)

    numerical_grads = numerical_gradient_3vars(fn, x, y, z)

    assert abs(analytical_grads["x"] - numerical_grads["x"]) < 1e-6
    assert abs(analytical_grads["y"] - numerical_grads["y"]) < 1e-6
    assert abs(analytical_grads["z"] - numerical_grads["z"]) < 1e-6
