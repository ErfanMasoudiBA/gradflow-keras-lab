from gradflow.value import Value
import math

def test_value_creation():
    x = Value(2.0)

    assert x.data == 2.0
    assert x.grad == 0.0


def test_value_add():
    x = Value(2.0)
    y = Value(3.0)

    z = x + y

    assert z.data == 5.0


def test_value_mul():
    x = Value(2.0)
    y = Value(3.0)

    z = x * y

    assert z.data == 6.0


def test_value_pow():
    x = Value(3.0)

    y = x ** 2

    assert y.data == 9.0


def test_value_backward_simple_add():
    x = Value(2.0)
    y = Value(3.0)

    z = x + y
    z.backward()

    assert x.grad == 1.0
    assert y.grad == 1.0
    assert z.grad == 1.0


def test_value_backward_simple_mul():
    x = Value(2.0)
    y = Value(3.0)

    z = x * y
    z.backward()

    assert x.grad == 3.0
    assert y.grad == 2.0
    assert z.grad == 1.0


def test_value_backward_xyz_graph():
    x = Value(2.0)
    y = Value(3.0)
    z = Value(4.0)

    L = ((x * y) + z) ** 2

    L.backward()

    assert L.data == 100.0
    assert x.grad == 60.0
    assert y.grad == 40.0
    assert z.grad == 20.0


def test_value_with_python_numbers():
    x = Value(2.0)

    y = x + 3
    z = 3 + x
    a = x * 4
    b = 4 * x

    assert y.data == 5.0
    assert z.data == 5.0
    assert a.data == 8.0
    assert b.data == 8.0

def test_value_gradient_accumulation_add():
    x = Value(3.0)

    y = x + x
    y.backward()

    assert y.data == 6.0
    assert x.grad == 2.0

def test_value_gradient_accumulation_mul():
    x = Value(3.0)

    y = x * x
    y.backward()

    assert y.data == 9.0
    assert x.grad == 6.0


def test_value_sub():
    x = Value(5.0)
    y = Value(2.0)

    z = x - y

    assert z.data == 3.0

def test_value_rsub():
    x = Value(2.0)

    z = 5.0 - x

    assert z.data == 3.0

def test_value_div():
    x = Value(8.0)
    y = Value(2.0)

    z = x / y

    assert z.data == 4.0

def test_value_rdiv():
    x = Value(2.0)

    z = 8.0 / x

    assert z.data == 4.0

def test_value_neg():
    x = Value(3.0)

    y = -x

    assert y.data == -3.0


def test_value_backward_sub():
    x = Value(5.0)
    y = Value(2.0)

    z = x - y
    z.backward()

    assert x.grad == 1.0
    assert y.grad == -1.0

def test_value_backward_div():
    x = Value(8.0)
    y = Value(2.0)

    z = x / y
    z.backward()

    assert abs(x.grad - 0.5) < 1e-9
    assert abs(y.grad - (-2.0)) < 1e-9

def test_value_tanh():
    x = Value(0.5)

    y = x.tanh()

    assert abs(y.data - math.tanh(0.5)) < 1e-9

def test_value_backward_tanh():
    x = Value(0.5)

    y = x.tanh()
    y.backward()

    expected = 1 - math.tanh(0.5) ** 2
    assert abs(x.grad - expected) < 1e-9

def test_value_relu_positive():
    x = Value(3.0)

    y = x.relu()

    assert y.data == 3.0

def test_value_relu_negative():
    x = Value(-3.0)

    y = x.relu()

    assert y.data == 0.0

def test_value_backward_relu_positive():
    x = Value(3.0)

    y = x.relu()
    y.backward()

    assert x.grad == 1.0

def test_value_backward_relu_negative():
    x = Value(-3.0)

    y = x.relu()
    y.backward()

    assert x.grad == 0.0

def test_value_zero_grad():
    x = Value(2.0)
    y = Value(3.0)

    z = x * y
    z.backward()

    assert x.grad == 3.0
    assert y.grad == 2.0

    x.zero_grad()
    y.zero_grad()

    assert x.grad == 0.0
    assert y.grad == 0.0
