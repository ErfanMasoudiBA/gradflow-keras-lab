import pytest

from gradflow.gates import (
    add_forward,
    add_backward,
    mul_forward,
    mul_backward,
    max_forward,
    max_backward,
    copy_forward,
    copy_backward,
)


def test_add_forward():
    assert add_forward(3, 5) == 8
    assert add_forward(-2, 7) == 5


def test_add_backward():
    dx, dy = add_backward(4)
    assert dx == 4
    assert dy == 4


def test_mul_forward():
    assert mul_forward(3, 5) == 15
    assert mul_forward(-2, 4) == -8


def test_mul_backward():
    x = 3
    y = 5
    upstream_grad = 2

    dx, dy = mul_backward(x, y, upstream_grad)

    assert dx == 10
    assert dy == 6


def test_max_forward():
    assert max_forward(3, 5) == 5
    assert max_forward(10, 4) == 10


def test_max_backward_when_x_wins():
    dx, dy = max_backward(x=10, y=4, upstream_grad=3)

    assert dx == 3
    assert dy == 0.0


def test_max_backward_when_y_wins():
    dx, dy = max_backward(x=3, y=5, upstream_grad=2)

    assert dx == 0.0
    assert dy == 2


def test_max_backward_tie_split():
    dx, dy = max_backward(x=5, y=5, upstream_grad=10, tie_mode="split")

    assert dx == 5
    assert dy == 5


def test_max_backward_tie_first():
    dx, dy = max_backward(x=5, y=5, upstream_grad=10, tie_mode="first")

    assert dx == 10
    assert dy == 0.0


def test_max_backward_tie_second():
    dx, dy = max_backward(x=5, y=5, upstream_grad=10, tie_mode="second")

    assert dx == 0.0
    assert dy == 10


def test_max_backward_invalid_tie_mode():
    with pytest.raises(ValueError):
        max_backward(x=5, y=5, upstream_grad=10, tie_mode="invalid")


def test_copy_forward():
    branches = copy_forward(7, n=3)

    assert branches == (7, 7, 7)


def test_copy_backward():
    dx = copy_backward(2, 1, -4)

    assert dx == -1
