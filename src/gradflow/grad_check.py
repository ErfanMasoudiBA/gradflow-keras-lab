def numerical_gradient_3vars(fn, x, y, z, h=1e-5):
    """
    Compute numerical gradients for a scalar function of three variables.

    fn should be a function like:
        fn(x, y, z) -> scalar
    """
    dx = (fn(x + h, y, z) - fn(x - h, y, z)) / (2 * h)
    dy = (fn(x, y + h, z) - fn(x, y - h, z)) / (2 * h)
    dz = (fn(x, y, z + h) - fn(x, y, z - h)) / (2 * h)

    return {
        "x": dx,
        "y": dy,
        "z": dz,
    }
