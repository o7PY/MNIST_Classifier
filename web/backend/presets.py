"""Famous architecture presets, expressed as node/edge graphs.

Every graph implicitly starts at the fixed `input` node (`Input`, 1x28x28)
and ends at the fixed `output` node (`Output`, auto Linear(->10), no
activation). Both are included explicitly here so presets are ordinary,
complete graphs - the same shape `graph.py` will validate in Phase 4.
"""

SIMPLE_MLP = {
    "id": "simple_mlp",
    "name": "Simple MLP",
    "description": "A single hidden layer dense network.",
    "nodes": [
        {"id": "input", "type": "Input", "config": {}},
        {"id": "flatten_1", "type": "Flatten", "config": {}},
        {"id": "linear_1", "type": "Linear", "config": {"out_features": 128}},
        {"id": "relu_1", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "output", "type": "Output", "config": {}},
    ],
    "edges": [
        {"from": "input", "to": "flatten_1"},
        {"from": "flatten_1", "to": "linear_1"},
        {"from": "linear_1", "to": "relu_1"},
        {"from": "relu_1", "to": "output"},
    ],
}

LENET5 = {
    "id": "lenet5",
    "name": "LeNet-5 (adapted)",
    "description": "Classic conv/pool stack adapted to 28x28 input.",
    "nodes": [
        {"id": "input", "type": "Input", "config": {}},
        {"id": "conv_1", "type": "Conv2d", "config": {"out_channels": 6, "kernel_size": 5, "stride": 1, "padding": 0}},
        {"id": "relu_1", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "pool_1", "type": "MaxPool2d", "config": {"kernel_size": 2}},
        {"id": "conv_2", "type": "Conv2d", "config": {"out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0}},
        {"id": "relu_2", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "pool_2", "type": "MaxPool2d", "config": {"kernel_size": 2}},
        {"id": "flatten_1", "type": "Flatten", "config": {}},
        {"id": "linear_1", "type": "Linear", "config": {"out_features": 120}},
        {"id": "relu_3", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "linear_2", "type": "Linear", "config": {"out_features": 84}},
        {"id": "relu_4", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "output", "type": "Output", "config": {}},
    ],
    "edges": [
        {"from": "input", "to": "conv_1"},
        {"from": "conv_1", "to": "relu_1"},
        {"from": "relu_1", "to": "pool_1"},
        {"from": "pool_1", "to": "conv_2"},
        {"from": "conv_2", "to": "relu_2"},
        {"from": "relu_2", "to": "pool_2"},
        {"from": "pool_2", "to": "flatten_1"},
        {"from": "flatten_1", "to": "linear_1"},
        {"from": "linear_1", "to": "relu_3"},
        {"from": "relu_3", "to": "linear_2"},
        {"from": "linear_2", "to": "relu_4"},
        {"from": "relu_4", "to": "output"},
    ],
}

SMALL_CNN = {
    "id": "small_cnn",
    "name": "Small CNN",
    "description": "Two conv/pool blocks with dropout before the classifier head.",
    "nodes": [
        {"id": "input", "type": "Input", "config": {}},
        {"id": "conv_1", "type": "Conv2d", "config": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 0}},
        {"id": "relu_1", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "pool_1", "type": "MaxPool2d", "config": {"kernel_size": 2}},
        {"id": "conv_2", "type": "Conv2d", "config": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
        {"id": "relu_2", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "pool_2", "type": "MaxPool2d", "config": {"kernel_size": 2}},
        {"id": "flatten_1", "type": "Flatten", "config": {}},
        {"id": "dropout_1", "type": "Dropout", "config": {"p": 0.5}},
        {"id": "linear_1", "type": "Linear", "config": {"out_features": 128}},
        {"id": "relu_3", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "output", "type": "Output", "config": {}},
    ],
    "edges": [
        {"from": "input", "to": "conv_1"},
        {"from": "conv_1", "to": "relu_1"},
        {"from": "relu_1", "to": "pool_1"},
        {"from": "pool_1", "to": "conv_2"},
        {"from": "conv_2", "to": "relu_2"},
        {"from": "relu_2", "to": "pool_2"},
        {"from": "pool_2", "to": "flatten_1"},
        {"from": "flatten_1", "to": "dropout_1"},
        {"from": "dropout_1", "to": "linear_1"},
        {"from": "linear_1", "to": "relu_3"},
        {"from": "relu_3", "to": "output"},
    ],
}

RESIDUAL_CNN = {
    "id": "residual_cnn",
    "name": "Residual CNN (skip connection)",
    "description": "A conv block with a skip connection merged via Add — exercises graph branching.",
    "nodes": [
        {"id": "input", "type": "Input", "config": {}},
        {"id": "conv_1", "type": "Conv2d", "config": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "relu_1", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "conv_2", "type": "Conv2d", "config": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "relu_2", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "conv_3", "type": "Conv2d", "config": {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
        {"id": "add_1", "type": "Add", "config": {}},
        {"id": "relu_3", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "pool_1", "type": "MaxPool2d", "config": {"kernel_size": 2}},
        {"id": "flatten_1", "type": "Flatten", "config": {}},
        {"id": "linear_1", "type": "Linear", "config": {"out_features": 128}},
        {"id": "relu_4", "type": "Activation", "config": {"type": "ReLU"}},
        {"id": "output", "type": "Output", "config": {}},
    ],
    "edges": [
        {"from": "input", "to": "conv_1"},
        {"from": "conv_1", "to": "relu_1"},
        # relu_1 is "x": one branch goes straight to the Add (identity/skip),
        # the other goes through two more conv layers first.
        {"from": "relu_1", "to": "conv_2"},
        {"from": "conv_2", "to": "relu_2"},
        {"from": "relu_2", "to": "conv_3"},
        {"from": "relu_1", "to": "add_1"},
        {"from": "conv_3", "to": "add_1"},
        {"from": "add_1", "to": "relu_3"},
        {"from": "relu_3", "to": "pool_1"},
        {"from": "pool_1", "to": "flatten_1"},
        {"from": "flatten_1", "to": "linear_1"},
        {"from": "linear_1", "to": "relu_4"},
        {"from": "relu_4", "to": "output"},
    ],
}

PRESETS = [SIMPLE_MLP, LENET5, SMALL_CNN, RESIDUAL_CNN]
