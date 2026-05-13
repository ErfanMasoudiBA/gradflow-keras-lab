from gradflow.value import Value


def main():
    x = Value(2.0)
    y = Value(3.0)
    z = Value(4.0)

    L = ((x * y) + z) ** 2

    L.backward()

    print("Expression:")
    print("  L = ((x * y) + z)^2")
    print()

    print("Forward:")
    print(f"  L = {L.data}")
    print()

    print("Backward:")
    print(f"  dL/dx = {x.grad}")
    print(f"  dL/dy = {y.grad}")
    print(f"  dL/dz = {z.grad}")


if __name__ == "__main__":
    main()
