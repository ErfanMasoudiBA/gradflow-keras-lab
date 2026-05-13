from gradflow.nn import Neuron
from gradflow.value import Value


def main():
    n = Neuron(3)

    x = [
        Value(1.0, label="x1"),
        Value(2.0, label="x2"),
        Value(3.0, label="x3"),
    ]

    y = n(x)

    print("Neuron output:")
    print(y)

    y.backward()

    print("\nInput gradients:")
    for xi in x:
        print(f"{xi.label}: {xi.grad}")

    print("\nParameter gradients:")
    for p in n.parameters():
        print(p.grad)


if __name__ == "__main__":
    main()
