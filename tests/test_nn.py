from gradflow.nn import Neuron, Layer
from gradflow.value import Value

def test_neuron_output_type():
    n = Neuron(3)

    x = [Value(1.0), Value(2.0), Value(3.0)]

    y = n(x)

    assert isinstance(y, Value)

def test_neuron_parameters_count():
    n = Neuron(3)

    params = n.parameters()

    assert len(params) == 4

def test_layer_output_count():
    layer = Layer(3, 4)

    x = [Value(1.0), Value(2.0), Value(3.0)]

    outputs = layer(x)

    assert len(outputs) == 4

def test_layer_parameters_count():
    layer = Layer(3, 4)

    params = layer.parameters()

    assert len(params) == 16
