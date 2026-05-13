from gradflow.simple_graphs import forward_xyz_graph, backward_xyz_graph


def main():
    x = 2
    y = 3
    z = 4

    L, cache = forward_xyz_graph(x, y, z)
    grads = backward_xyz_graph(cache)

    print("Graph:")
    print("  L = ((x * y) + z)^2")
    print()

    print("Inputs:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    print(f"  z = {z}")
    print()

    print("Forward:")
    print(f"  a = x * y = {cache['a']}")
    print(f"  b = a + z = {cache['b']}")
    print(f"  L = b^2 = {L}")
    print()

    print("Backward:")
    print(f"  dL/dx = {grads['x']}")
    print(f"  dL/dy = {grads['y']}")
    print(f"  dL/dz = {grads['z']}")


if __name__ == "__main__":
    main()
