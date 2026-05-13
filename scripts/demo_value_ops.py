from gradflow.value import Value


def main():
    x = Value(2.0, label="x")
    y = Value(-3.0, label="y")

    z = x - y
    q = x / 2
    r = z.tanh()
    s = y.relu()

    out = r + s + q
    out.backward()

    print("Values:")
    print(f"x = {x}")
    print(f"y = {y}")
    print()

    print("Forward:")
    print(f"z = x - y = {z.data}")
    print(f"q = x / 2 = {q.data}")
    print(f"r = tanh(z) = {r.data}")
    print(f"s = relu(y) = {s.data}")
    print(f"out = r + s + q = {out.data}")
    print()

    print("Gradients:")
    print(f"dout/dx = {x.grad}")
    print(f"dout/dy = {y.grad}")


if __name__ == "__main__":
    main()
