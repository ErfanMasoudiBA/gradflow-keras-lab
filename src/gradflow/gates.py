"""
Basic computational gates for learning gradient flow.

Each gate has:
- a forward function
- a backward function

The goal is educational clarity, not performance.
"""


def add_forward(x, y):
    """Forward pass for z = x + y."""
    return x + y


def add_backward(upstream_grad):
    """
    Backward pass for z = x + y.

    dz/dx = 1
    dz/dy = 1

    Therefore:
    dL/dx = dL/dz * dz/dx = upstream_grad
    dL/dy = dL/dz * dz/dy = upstream_grad
    """
    dx = upstream_grad
    dy = upstream_grad
    return dx, dy


def mul_forward(x, y):
    """Forward pass for z = x * y."""
    return x * y


def mul_backward(x, y, upstream_grad):
    """
    Backward pass for z = x * y.

    dz/dx = y
    dz/dy = x

    Therefore:
    dL/dx = upstream_grad * y
    dL/dy = upstream_grad * x
    """
    dx = y * upstream_grad
    dy = x * upstream_grad
    return dx, dy


def max_forward(x, y):
    """Forward pass for z = max(x, y)."""
    return max(x, y)


def max_backward(x, y, upstream_grad, tie_mode="split"):
    """
    Backward pass for z = max(x, y).

    The upstream gradient is routed to the input that won during forward.

    tie_mode controls what happens when x == y:
    - "split": split gradient equally between x and y
    - "first": send gradient to x
    - "second": send gradient to y
    """
    if x > y:
        return upstream_grad, 0.0

    if y > x:
        return 0.0, upstream_grad

    # Tie case: x == y
    if tie_mode == "split":
        return upstream_grad / 2, upstream_grad / 2

    if tie_mode == "first":
        return upstream_grad, 0.0

    if tie_mode == "second":
        return 0.0, upstream_grad

    raise ValueError("tie_mode must be one of: 'split', 'first', 'second'")


def copy_forward(x, n=2):
    """
    Forward pass for copying x into n branches.

    Example:
        copy_forward(3, n=2) -> (3, 3)
    """
    return tuple(x for _ in range(n))


def copy_backward(*branch_grads):
    """
    Backward pass for a value used in multiple branches.

    Gradients from all branches are summed.
    """
    return sum(branch_grads)
