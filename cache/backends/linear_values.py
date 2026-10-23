"""Closed-form key-to-value prediction using the MLP cache's residual storage."""

import torch

from .mlp_values import MLPValueCache, MLPValueLayer


class LinearPredictor(torch.nn.Module):
    def __init__(self, num_heads, head_dim, *, device, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(num_heads, head_dim, head_dim, device=device, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, keys):
        return torch.einsum("bhtd,hde->bhte", keys, self.weight)


class LinearValueLayer(MLPValueLayer):
    def lazy_initialization(self, value_states):
        if self.mlp is None:
            self.mlp = LinearPredictor(
                value_states.size(1),
                value_states.size(-1),
                device=value_states.device,
                dtype=value_states.dtype,
            )
            self._num_params = self.mlp.weight.numel()
        super().lazy_initialization(value_states)

    @torch.no_grad()
    def train_mlp(self, keys, padding_mask):
        # Pool valid tokens across the batch, independently for each KV head.
        inputs = keys.permute(1, 0, 2, 3)[:, padding_mask].float()
        targets = self.tensor.permute(1, 0, 2, 3)[:, padding_mask].float()
        # XKV can produce rank-deficient keys; pinv also handles that case.
        self.mlp.weight.copy_(torch.linalg.pinv(inputs) @ targets)


class LinearValueCache(MLPValueCache):
    def _build_layer(self, layer_idx):
        return LinearValueLayer(
            target_cr=self.target_cr,
            rope_cache=self.rope_cache,
            turboquant_residuals=self.turboquant_residuals,
            compressor_bits=self.compressor_bits,
            residual_sink_tokens=self.residual_sink_tokens,
        )
