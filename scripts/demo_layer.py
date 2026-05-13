from gradflow.nn import Layer
from gradflow.value import Value


def main():
    layer = Layer(3, 4)

    x = [
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]

    outputs = layer(x)

    print("Layer outputs:")

    for i, out in enumerate(outputs):
        print(f"Neuron {i}: {out}")


if __name__ == "__main__":
    main()
