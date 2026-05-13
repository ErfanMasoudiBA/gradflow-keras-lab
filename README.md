# GradFlow

GradFlow is a tiny deep learning engine built from scratch in Python.

The goal of this project is to understand how automatic differentiation, backpropagation, and neural network building blocks work under the hood — without relying on PyTorch, TensorFlow, or other large frameworks.

At its current stage, GradFlow includes:

- a scalar-based autograd engine
- computational graph construction
- reverse-mode automatic differentiation (`backward`)
- basic mathematical operations
- activation functions
- neural network primitives such as `Neuron` and `Layer`

This project is inspired by educational deep learning implementations like micrograd, but is being built step by step as a learning-focused framework.

---

## Features

### Implemented

- Scalar `Value` object with:
  - `data`
  - `grad`
  - graph tracking
  - backward propagation
- Basic operations:
  - addition
  - multiplication
  - subtraction
  - division
  - power
  - negation
- Nonlinearities:
  - `tanh`
  - `exp`
- Topological graph traversal for backprop
- `Neuron` abstraction
- `Layer` abstraction
- Parameter collection via `parameters()`

### In Progress

- `MLP` (multi-layer perceptron)
- loss functions
- training loop
- toy dataset experiments
- visualization utilities

---

## Why this project exists

Modern deep learning libraries are extremely powerful, but they often hide the mechanics behind:

- how gradients are computed
- how computation graphs are built
- how backpropagation flows through operations
- how neural network parameters are updated

GradFlow exists to make those ideas explicit and understandable.

This is not just a library project — it is a systems-level learning project.

---

## Project Structure

````text
.
├── src/
│   └── gradflow/
│       ├── __init__.py
│       ├── value.py
│       └── nn.py
├── scripts/
│   ├── demo_neuron.py
│   └── demo_layer.py
├── tests/
│   ├── test_value.py
│   └── test_nn.py
└── README.md
‍‍```
---

## Core Concept

GradFlow currently works with a scalar-based value object.

Example:

```python
from gradflow.value import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b
d = c + a
d.backward()

print(d.data)
print(a.grad)
print(b.grad)
````

Each `Value` tracks:

- its numerical data
- the operation that created it
- the previous nodes in the graph
- its gradient during backpropagation

---

## Neural Network Primitives

A single neuron can be created like this:

```python
from gradflow.nn import Neuron
from gradflow.value import Value

n = Neuron(3)
x = [Value(1.0), Value(2.0), Value(3.0)]

y = n(x)
print(y)

y.backward()

for p in n.parameters():
print(p.grad)
```

A layer of neurons:

```python
from gradflow.nn import Layer
from gradflow.value import Value

layer = Layer(3, 4)
x = [Value(1.0), Value(2.0), Value(3.0)]

out = layer(x)
print(out)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gradflow.git
cd gradflow
```

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
pip install -e .
```

Install test dependencies if needed:

```bash
pip install pytest
```

---

## Running Demos

Example neuron demo:

```bash
python scripts/demo_neuron.py
```

Example layer demo:

```bash
python scripts/demo_layer.py
```

---

## Running Tests

```bash
pytest
```

---

## Current Limitations

GradFlow is intentionally minimal right now.

Current limitations include:

- scalar-only autograd (no tensor support yet)
- no optimizer abstraction yet
- no training loop yet
- no dataset pipeline yet
- no GPU or NumPy backend yet

These are planned as future steps.

---

## Roadmap

Planned next phases:

1. Build `MLP`
2. Add loss functions
3. Implement training loop
4. Train on a toy dataset like XOR
5. Add prediction/evaluation utilities
6. Add graph visualization
7. Refactor into a cleaner framework structure
8. Explore optimizers like SGD / Adam
9. Potential tensor support in future versions

---

## Educational Goals

This project is being built to deeply understand:

- reverse-mode autodiff
- computational graphs
- gradient flow
- parameterized models
- neural network training from first principles

If you can build this yourself, you understand deep learning at a much more fundamental level than someone who only imports a framework.

---

## Inspiration

This project is inspired by the idea that the best way to understand deep learning systems is to build them from scratch.
