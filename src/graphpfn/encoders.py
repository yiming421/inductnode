"""
Encoder modules for GraphPFN.

Adapted from TabPFN's encoder implementations with simplified logic.
Includes NaN-aware statistics and encoding steps.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


# ============================================================================
# NaN-aware statistics functions (from TabPFN)
# ============================================================================

def torch_nansum(
    x: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the sum of a tensor, treating NaNs as zero.

    Args:
        x: The input tensor.
        axis: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `axis` retained or not.

    Returns:
        The sum of the tensor with NaNs treated as zero.
    """
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim)


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    include_inf: bool = False,
) -> torch.Tensor:
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs.
    """
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Computes the standard deviation of a tensor over a given dimension, ignoring NaNs.

    This implementation is designed for stability. It clips the denominator `(num - 1)`
    at a minimum of 1 to prevent division-by-zero errors.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.

    Returns:
        The standard deviation of the input tensor, ignoring NaNs.
    """
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num.clip(min=1.0)
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    # Clip the denominator to avoid division by zero when num=1
    var = torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1).clip(
        min=1.0
    )
    return torch.sqrt(var)


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    clip: bool = True,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize data to mean 0 and std 1 with high numerical stability.

    This function is designed to be robust against several edge cases:
    1. Constant Features: If a feature is constant, std=0 is replaced with 1.
    2. Single-Sample Normalization: If normalizing a single sample, std is set to 1.
    3. Low-Precision Dtypes: Adds epsilon (1e-16) to prevent overflow to infinity.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions`
            positions for normalization.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.

    Returns:
        The normalized data tensor.
    """
    assert (mean is None) == (std is None), (
        "Either both or none of mean and std must be given"
    )

    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)
            std = torch_nanstd(data[:normalize_positions], axis=0)
        else:
            mean = torch_nanmean(data, axis=0)
            std = torch_nanstd(data, axis=0)

        # Replace std=0 with 1 to avoid division by zero
        std = torch.where(std == 0, torch.ones_like(std), std)

        if len(data) == 1 or normalize_positions == 1:
            std = torch.ones_like(std)

    # Add epsilon for numerical stability
    data = (data - mean) / (std + 1e-16)

    if clip:
        data = torch.clip(data, min=-100, max=100)

    return data


# ============================================================================
# Encoder base classes (from TabPFN)
# ============================================================================

class SeqEncStep(nn.Module):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper that defines expected input keys and produced output keys.
    Subclasses should implement `_fit` and `_transform`.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the SeqEncStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
        """Fit the encoder step on the training set.

        Args:
            *x: The input tensors.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    def _transform(
        self,
        *x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, ...]:
        """Transform the data using the fitted encoder step.

        Args:
            *x: The input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            The transformed output tensor or tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self,
        state: dict,
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Perform the forward pass of the encoder step.

        Args:
            state: The input state dictionary containing the input tensors.
            cache_trainset_representation: Whether to cache the training set representation.
            **kwargs: Additional keyword arguments.

        Returns:
            The updated state dictionary with output tensors.
        """
        args = [state[in_key] for in_key in self.in_keys]

        # Fit only if needed
        if kwargs.get("single_eval_pos") or not cache_trainset_representation:
            self._fit(*args, **kwargs)

        out = self._transform(*args, **kwargs)

        assert isinstance(out, tuple), f"out must be a tuple, got {type(out)}"
        assert len(out) == len(self.out_keys), \
            f"Expected {len(self.out_keys)} outputs, got {len(out)}"

        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state


class SequentialEncoder(nn.Sequential):
    """An encoder that applies a sequence of encoder steps.

    SequentialEncoder allows building an encoder from a sequence of SeqEncStep instances.
    The input is passed through each step in the provided order.
    """

    def __init__(self, *args: SeqEncStep, output_key: str = "output", **kwargs: Any):
        """Initialize the SequentialEncoder.

        Args:
            *args: A list of SeqEncStep instances to apply in order.
            output_key: The key to use for the output. Defaults to "output".
            **kwargs: Additional keyword arguments passed to nn.Sequential.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input: dict, **kwargs: Any) -> dict:
        """Apply the sequence of encoder steps to the input.

        Args:
            input: The input state dictionary.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            The output of the final encoder step.
        """
        # If the input is not a dict and the first layer expects one input,
        # map the input to the first input key
        if not isinstance(input, dict) and len(self[0].in_keys) == 1:
            input = {self[0].in_keys[0]: input}

        for module in self:
            input = module(input, **kwargs)

        return input[self.output_key] if self.output_key is not None else input


# ============================================================================
# Concrete encoder implementations
# ============================================================================

class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""
        pass

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)

        # Ensure input tensor dtype and device match the layer's weight
        x = x.to(dtype=self.layer.weight.dtype, device=self.layer.weight.device)

        return (self.layer(x),)


class FourierFeatureEncoderStep(SeqEncStep):
    """
    Element-wise Random Fourier Feature encoder for coordinate-free learning.

    Unlike standard RFF, this encodes each feature INDEPENDENTLY without mixing:
        For each feature i: output_i = [cos(2π * B * x_i), sin(2π * B * x_i)]

    This maintains permutation invariance - features can be reordered without
    changing the embedding of any individual value. Critical for transfer learning
    where column semantics change.

    Output shape: [batch, num_features, emsize]
    (Ready to be fed into a Set Transformer as a sequence of tokens)
    """

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        scale: float = 1.0,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the FourierFeatureEncoderStep.

        Args:
            num_features: The number of input features (used for validation).
            emsize: The embedding size per feature (must be even, split into cos/sin).
            replace_nan_by_zero: Whether to replace NaN values by zero.
            scale: Standard deviation of the random Fourier feature frequencies.
                   Higher scale = sensitive to fine-grained value differences.
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__(in_keys, out_keys)
        assert emsize % 2 == 0, f"emsize must be even for Fourier features, got {emsize}"

        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero
        self.scale = scale

        # Random frequencies for Fourier features (fixed, not learned)
        # Shape: [emsize//2]
        # We share the SAME frequencies for all features to ensure
        # consistent encoding across dimensions.
        self.register_buffer(
            "B",
            torch.randn(emsize // 2) * scale
        )

    def _fit(self, *x: torch.Tensor, **kwargs: Any) -> None:
        """Fit the encoder step. Does nothing for FourierFeatureEncoderStep."""
        pass

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply element-wise Fourier feature transformation.

        Args:
            *x: The input tensors to concatenate and transform.
                Expected shape: [Batch, Num_Features] or [Batch, ..., Num_Features]
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor of shape [..., Num_Features, emsize].
        """
        x = torch.cat(x, dim=-1)

        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)

        # Ensure input tensor dtype and device match B
        x = x.to(dtype=self.B.dtype, device=self.B.device)

        # --- CRITICAL LOGIC FOR COORDINATE INDEPENDENCE ---

        # Compute 2π * B * x
        # x: [..., num_features] where num_features=1 for this encoder
        # B: [emsize//2]
        # We want: [..., emsize//2] (broadcast B across batch dimensions)

        # For input shape [..., 1], multiply element-wise with B
        # Result: [..., 1] * [emsize//2] broadcasts to [..., emsize//2]
        x_proj = x * self.B * (2.0 * torch.pi)

        # Apply Cos/Sin and Concatenate
        # Output: [..., emsize]
        output = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        return (output,)


class NanHandlingEncoderStep(SeqEncStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Args:
            keep_nans: Whether to keep NaN values as separate indicators. Defaults to True.
            in_keys: The keys of the input tensors. Must be a single key.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects a single input key"
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the feature means on the training set for replacing NaNs.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        self.feature_means_ = torch_nanmean(
            x[:single_eval_pos],
            axis=0,
            include_inf=True,
        )

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replace NaN and infinite values in the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        nans_indicator = None
        if self.keep_nans:
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # Replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]
        return x, nans_indicator
