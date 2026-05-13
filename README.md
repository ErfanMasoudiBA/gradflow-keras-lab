# gradflow-keras-lab

An educational project for understanding gradient flow, backpropagation, and later deep learning workflows with Keras.

The goal is not to build a high-performance framework.  
The goal is to understand what really happens behind automatic differentiation.

## Phase 1: Gradient Gates

In Phase 1, we implemented simple computational gates:

- Add Gate
- Multiply Gate
- Max Gate
- Copy Gate

Each gate has:

- forward pass
- backward pass

These gates help us understand how gradients move backward through a computational graph.

## Project Structure

````text
gradflow-keras-lab/
├── src/
│   └── gradflow/
│       ├── __init__.py
│       └── gates.py
├── scripts/
│   └── demo_gates.py
├── tests/
│   └── test_gates.py
├── README.md
└── requirements.txt

## Run Demo

Linux / macOS:

```bash
PYTHONPATH=src python scripts/demo_gates.py
````

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/demo_gates.py
```

## Run Tests

Linux / macOS:

```bash
PYTHONPATH=src pytest
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
pytest
```

## Phase 2 Preview

In Phase 2, we will build a small computational graph manually:

‍‍‍‍‍‍‍‍‍```text
L = ((x \* y) + z)^2

```

We will implement:

- forward pass
- backward pass
- manual chain rule
- numerical gradient checking


```
