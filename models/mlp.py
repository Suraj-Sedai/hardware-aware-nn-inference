# models/mlp.py

from typing import List, Callable
from core.tensor import Tensor
from kernels.linear_fp32 import linear_fp32
import numpy as np


def relu(tensor: Tensor) -> Tensor:
    """Elementwise ReLU activation."""
    if tensor.dtype != Tensor._dtype:
        # For now we know dtype is FP32
        pass
    return Tensor.from_fp32(np.maximum(tensor.data, 0))


class MLP:
    """
    Simple FP32 Multi-Layer Perceptron.
    Forward pass only.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Callable = relu,
    ):
        """
        Initialize MLP with given layer sizes.

        layer_sizes: [in_features, hidden1, hidden2, ..., out_features]
        """
        self.num_layers = len(layer_sizes) - 1
        self.layers = []

        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            # Initialize weight and bias
            weight_array = np.random.randn(out_dim, in_dim).astype(np.float32)
            bias_array = np.random.randn(out_dim).astype(np.float32)

            weight_tensor = Tensor.from_fp32(weight_array)
            bias_tensor = Tensor.from_fp32(bias_array)

            # Store layer info
            self.layers.append({
                "weight": weight_tensor,
                "bias": bias_tensor,
                "activation": activation if i < self.num_layers - 1 else None
            })

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.
        """
        out = x
        for layer in self.layers:
            out = linear_fp32(out, layer["weight"], layer["bias"])
            if layer["activation"] is not None:
                out = layer["activation"](out)
        return out
