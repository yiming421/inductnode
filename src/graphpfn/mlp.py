"""
MLP module for GraphPFN.

Adapted from TabPFN's MLP implementation.
"""

from __future__ import annotations

from enum import Enum

import torch


class Activation(Enum):
    """Enum for activation functions."""

    GELU = 1
    RELU = 2


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module.

    This module consists of two linear layers with an activation function in between.
    Simplified version of TabPFN's MLP without memory optimization decorators.

    Args:
        size: The input and output size of the MLP.
        hidden_size: The size of the hidden layer.
        activation: The activation function to use ('gelu' or 'relu').
        device: The device to use for the linear layers.
        dtype: The data type to use for the linear layers.

    Example:
        >>> mlp = MLP(size=128, hidden_size=512, activation='gelu')
        >>> x = torch.randn(32, 128)
        >>> output = mlp(x)
    """

    def __init__(
        self,
        size: int,
        hidden_size: int,
        activation: Activation | str = "gelu",
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear2 = torch.nn.Linear(
            hidden_size,
            size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        *,
        add_input: bool = False,
    ) -> torch.Tensor:
        """Performs the forward pass of the MLP.

        Args:
            x: The input tensor.
            add_input: Whether to add input to the output (residual). Default is False.

        Returns:
            The output tensor.
        """
        input_shape = x.shape
        input_tensor = x

        # Flatten to 2D for linear layers
        x = x.reshape(-1, x.size(-1))

        # First linear + activation
        x = self.linear1(x)
        if self.activation is Activation.GELU:
            x = torch.nn.functional.gelu(x)
        elif self.activation is Activation.RELU:
            x = torch.nn.functional.relu(x)
        else:
            raise NotImplementedError(
                f"Activation Function {self.activation} is not implemented.",
            )

        # Second linear
        x = self.linear2(x)

        # Reshape back
        x = x.reshape(input_shape)

        # Optional residual connection
        if add_input:
            # DEBUG: Check magnitude ratio
            input_norm = input_tensor.norm().item()
            mlp_norm = x.norm().item()
            ratio = input_norm / (mlp_norm + 1e-9)
            if ratio > 5.0 or ratio < 0.2:  # Log if imbalanced
                print(f"[MLP MAGNITUDE] Input norm: {input_norm:.4f}, MLP output norm: {mlp_norm:.4f}, Ratio: {ratio:.2f}x")

            x = x + input_tensor

        return x
