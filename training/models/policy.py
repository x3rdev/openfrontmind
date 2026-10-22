import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

SIZE_MIN, SIZE_MAX = 0.01, 0.999
RESERVE_FLOOR = float(os.environ.get("OFM_RESERVE_FLOOR", "0.2"))
CONV_OUT_CHANNELS = 32
POOL_SIZE = 16


class EntityEncoder(nn.Module):
    """Per-entity MLP plus attention pooling. Works for any number of entities."""

    def __init__(self, entity_dim: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.pool_query = nn.Parameter(torch.randn(embed_dim) * embed_dim ** -0.5)

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """entities (B, N, entity_dim), mask (B, N) True for real rows.
        Returns the pooled context (B, embed_dim) and per-entity embeddings (B, N, embed_dim)."""
        embeds = self.mlp(entities)
        scores = (embeds @ self.pool_query) / self.embed_dim ** 0.5
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * embeds).sum(dim=1)
        return pooled, embeds


class Backbone(nn.Module):
    """CNN over the map, entity encoder, and a merge MLP producing one hidden vector."""

    def __init__(self, spatial_channels: int, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, CONV_OUT_CHANNELS, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((POOL_SIZE, POOL_SIZE))
        self.entity_encoder = EntityEncoder(entity_dim, entity_embed_dim)

        pooled_size = CONV_OUT_CHANNELS * POOL_SIZE * POOL_SIZE
        self.merge_mlp = nn.Sequential(
            nn.Linear(pooled_size + scalar_dim + entity_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

    def forward(
        self, spatial: torch.Tensor, scalar: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_conv = self.pool(self.conv_stack(spatial)).flatten(start_dim=1)
        entity_context, entity_embeds = self.entity_encoder(entities, entity_mask)
        combined = torch.cat([pooled_conv, scalar, entity_context], dim=1)
        return self.merge_mlp(combined), entity_embeds


class Agent(nn.Module):
    """Each tick: a gate (wait or act), a pointer over candidate targets, and a Beta
    attack size. The action is encoded as target_index + size_fraction in one float."""

    def __init__(self, spatial_channels: int, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.entity_embed_dim = entity_embed_dim
        self.backbone = Backbone(spatial_channels, scalar_dim, entity_dim, entity_embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(256 + scalar_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.query_proj = nn.Linear(256, entity_embed_dim)
        self.gate_head = nn.Linear(256 + scalar_dim, 2)  # logits: [wait, act]
        self.size_head = nn.Linear(256 + scalar_dim + entity_embed_dim, 2)
        nn.init.zeros_(self.size_head.weight)
        with torch.no_grad():
            self.size_head.bias.copy_(torch.tensor([0.5413, 6.0025]))  # starts as Beta(2, 7)
        self.reserve_floor = RESERVE_FLOOR
        self.gate_kl = None

    @classmethod
    def from_observation(cls, spatial: np.ndarray, scalar: np.ndarray, entities: np.ndarray) -> "Agent":
        return cls(spatial_channels=spatial.shape[0], scalar_dim=scalar.shape[0], entity_dim=entities.shape[-1])

    def get_value(
        self, spatial: torch.Tensor, scalar: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden, _ = self.backbone(spatial, scalar, entities, entity_mask)
        return self.value_head(torch.cat([hidden, scalar], dim=-1))

    def soft_update_from(self, source: "Agent", tau: float) -> None:
        """EMA update toward the source's weights."""
        with torch.no_grad():
            for target_param, source_param in zip(self.parameters(), source.parameters()):
                target_param.mul_(1 - tau).add_(source_param, alpha=tau)

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        return_probs: bool = False,
        return_entropy_components: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Samples an action when `action` is None, otherwise scores the given one.
        Returns (action, log_prob, entropy, value), plus optional extras."""
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)
        troop_ratio = scalar[:, 1]

        is_do_nothing = entities[..., 3] > 0.5
        is_already_attacking = entities[..., 8] > 0.5
        real_target_mask = entity_mask & ~is_do_nothing & ~is_already_attacking
        can_act = real_target_mask.any(dim=-1) & (troop_ratio >= self.reserve_floor)
        do_nothing_idx = is_do_nothing.float().argmax(dim=-1)

        query = self.query_proj(hidden)
        target_scores = torch.einsum("bnd,bd->bn", entity_embeds, query) / self.entity_embed_dim ** 0.5
        target_scores = target_scores.masked_fill(~real_target_mask, -1e8)
        target_dist = Categorical(logits=target_scores)

        gate_logits = self.gate_head(torch.cat([hidden, scalar], dim=-1))
        act_logit = torch.where(can_act, gate_logits[:, 1], torch.full_like(gate_logits[:, 1], -1e8))
        gate_dist = Categorical(logits=torch.stack([gate_logits[:, 0], act_logit], dim=-1))

        # KL to a patient reference gate: almost never act below ~0.45 troop ratio
        ref_act = 0.005 + 0.495 * torch.sigmoid((troop_ratio - 0.5) / 0.03)
        ref_logp = torch.log(torch.stack([1 - ref_act, ref_act], dim=-1))
        self.gate_kl = (gate_dist.probs * (gate_dist.logits - ref_logp)).sum(dim=-1)

        if action is None:
            gate_action = gate_dist.sample()
            target_action = target_dist.sample()
            action = torch.where(gate_action == torch.tensor(1), target_action, do_nothing_idx)
            size_dist = self._size_dist(hidden, scalar, entity_embeds, action)
            size_action = size_dist.sample().clamp(SIZE_MIN, SIZE_MAX)
            size_action = torch.where(action == do_nothing_idx, torch.full_like(size_action, 0.5), size_action)
            action_out = action.double() + size_action.double()
        else:
            action_out = action
            target_idx = torch.floor(action).long()
            size_action = (action - target_idx.to(action.dtype)).float().clamp(SIZE_MIN, SIZE_MAX)
            action = target_idx
            size_dist = self._size_dist(hidden, scalar, entity_embeds, action)

        acted = (action != do_nothing_idx).long()
        log_prob = gate_dist.log_prob(acted) + acted.float() * (
            target_dist.log_prob(action) + size_dist.log_prob(size_action)
        )

        # gate entropy is left out; the gate is regularised through gate_kl instead
        num_candidates = real_target_mask.sum(dim=-1).float()
        target_entropy_ceiling = torch.where(num_candidates > 1, torch.log(num_candidates), torch.ones_like(num_candidates))
        normalized_target_entropy = target_dist.entropy() / target_entropy_ceiling
        p_act = gate_dist.probs[:, 1]
        entropy = p_act * normalized_target_entropy + p_act.detach() * size_dist.entropy()

        result = (action_out, log_prob, entropy, self.value_head(torch.cat([hidden, scalar], dim=-1)))
        if return_probs:
            # p(do_nothing) = p(wait), p(candidate j) = p(act) * p(target = j | act)
            flat_probs = gate_dist.probs[:, 1:2] * target_dist.probs
            flat_probs = torch.where(is_do_nothing, gate_dist.probs[:, 0:1].expand_as(flat_probs), flat_probs)
            result = result + (flat_probs,)
        if return_entropy_components:
            result = result + (gate_dist.entropy(), target_dist.entropy(), num_candidates)
        return result

    def _size_dist(self, hidden, scalar, entity_embeds, target_idx):
        emb = entity_embeds[torch.arange(entity_embeds.shape[0], device=entity_embeds.device), target_idx]
        raw = self.size_head(torch.cat([hidden, scalar, emb], dim=-1))
        return Beta(1.0 + F.softplus(raw[:, 0]), 1.0 + F.softplus(raw[:, 1]))
