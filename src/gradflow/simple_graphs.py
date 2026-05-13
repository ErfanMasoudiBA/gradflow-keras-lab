from gradflow.gates import (
    mul_forward,
    mul_backward,
    add_forward,
    add_backward,
    square_forward,
    square_backward,
)


def forward_xyz_graph(x, y, z):
    """
    Forward pass for:

        L = ((x * y) + z)^2

    Returns:
        L: final output
        cache: intermediate values needed for backward
    """
    a = mul_forward(x, y)
    b = add_forward(a, z)
    L = square_forward(b)

    cache = {
        "x": x,
        "y": y,
        "z": z,
        "a": a,
        "b": b,
        "L": L,
    }

    return L, cache


def backward_xyz_graph(cache):
    """
    Backward pass for:

        L = ((x * y) + z)^2

    Returns:
        gradients as a dictionary:
        {
            "x": dx,
            "y": dy,
            "z": dz
        }
    """
    x = cache["x"]
    y = cache["y"]
    a = cache["a"]
    b = cache["b"]

    # L = b^2
    dL = 1.0
    db = square_backward(b, dL)

    # b = a + z
    da, dz = add_backward(db)

    # a = x * y
    dx, dy = mul_backward(x, y, da)

    return {
        "x": dx,
        "y": dy,
        "z": dz,
    }
