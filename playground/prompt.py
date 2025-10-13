#%%
"""
Prompt pool module inspired by "Learning to Prompt for Continual Learning" (L2P).

This version assumes **2D inputs** (no channel dimension). You can prepend prompt vectors
directly along the feature dimension of your VAE input (e.g. if inputs are shaped `(B, D)`)
by concatenating the selected prompt vectors to the input features.

Usage pattern:
 1. Create a PromptPool:
      prompt_pool = PromptPool(num_prompts=100, prompt_dim=16, key_dim=32, top_k=5)

 2. For each input get a `query` vector with shape (batch, key_dim). The easiest is to
    provide the VAE latent z or an encoding of the input.

 3. Get prompt tensor:
      prompts = prompt_pool(query)  # shape (B, prompt_length, prompt_dim)

 4. To prepend to a 2D input X (B, D), use the helper:
      X_prepended = prepend_prompts_to_vector(X, prompts)

"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptPool(nn.Module):
    def __init__(
        self,
        num_prompts: int = 100,
        prompt_dim: int = 16,
        prompt_length: int = 1,
        key_dim: int = 32,
        top_k: int = 5,
        selection: str = "soft",
        temperature: float = 1.0,
        normalize_keys: bool = True,
        query_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert selection in ("soft", "hard"), "selection must be 'soft' or 'hard'"
        assert top_k >= 1 and top_k <= num_prompts

        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.prompt_length = prompt_length
        self.key_dim = key_dim
        self.top_k = top_k
        self.selection = selection
        self.temperature = temperature
        self.normalize_keys = normalize_keys

        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_length, prompt_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(num_prompts, key_dim) * 0.02)

        self.query_encoder = query_encoder
        self._has_query_proj = query_encoder is not None

    def compute_similarity(self, queries: torch.Tensor) -> torch.Tensor:
        keys = self.keys
        if self.normalize_keys:
            q = F.normalize(queries, dim=-1)
            k = F.normalize(keys, dim=-1)
            sim = torch.matmul(q, k.t())
        else:
            q = queries.unsqueeze(1)
            k = keys.unsqueeze(0)
            dist2 = torch.sum((q - k) ** 2, dim=-1)
            sim = -dist2
        return sim

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if self._has_query_proj and not (query.ndim == 2 and query.size(1) == self.key_dim):
            query = self.query_encoder(query)

        assert query.ndim == 2 and query.size(1) == self.key_dim, (
            f"query must be (B, {self.key_dim}) when no query_encoder provided"
        )

        sim = self.compute_similarity(query)
        B, N = sim.size()

        if self.top_k < N:
            topk_vals, topk_idx = torch.topk(sim, self.top_k, dim=-1)
            mask = torch.full_like(sim, float('-inf'))
            mask.scatter_(1, topk_idx, topk_vals)
            logits = mask / (self.temperature + 1e-8)
            weights = F.softmax(logits, dim=-1)
        else:
            logits = sim / (self.temperature + 1e-8)
            weights = F.softmax(logits, dim=-1)

        if self.selection == 'hard':
            _, argmax = torch.max(weights, dim=-1)
            hard_weights = torch.zeros_like(weights)
            hard_weights.scatter_(1, argmax.unsqueeze(1), 1.0)
            weights = hard_weights

        w = weights.unsqueeze(-1).unsqueeze(-1)
        P = self.prompts.unsqueeze(0)
        weighted = w * P
        summed = weighted.sum(dim=1)  # (B, L, D_p)

        return summed


def prepend_prompts_to_vector(x: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
    """Prepend prompt vectors to a 2D input.

    Args:
        x: (B, D)
        prompts: (B, L, D_p)
    Returns:
        (B, D + L*D_p)
    """
    assert x.ndim == 2, "x must be (B, D)"
    B, D = x.shape
    Bp, L, Dp = prompts.shape
    assert B == Bp, "batch size mismatch"

    prompts_flat = prompts.reshape(B, L * Dp)
    x_prepended = torch.cat([prompts_flat, x], dim=-1)
    return x_prepended


if __name__ == "__main__":
    pool = PromptPool(num_prompts=10, prompt_dim=8, prompt_length=2, key_dim=16, top_k=3)
    q = torch.randn(5, 16)
    prompts = pool(q)
    print('Prompts shape:', prompts.shape)  # (5, 2, 8)

    x = torch.randn(5, 32)
    x_prep = prepend_prompts_to_vector(x, prompts)
    print('Prepended shape:', x_prep.shape)  # (5, 32 + 16)

    print('2D PromptPool module test passed.')

# %%
