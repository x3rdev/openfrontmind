import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

Distribution.set_default_validate_args(False)

SIZE_BINS = (0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6)
SIZE_INIT_PROBS = (0.25, 0.25, 0.2, 0.15, 0.1, 0.04, 0.01)
ABLATE = set(os.environ.get("OFM_ABLATE", "").split(","))  # temporary test switch
RESERVE_FLOOR = float(os.environ.get("OFM_RESERVE_FLOOR", "0.2"))
CONV_OUT_CHANNELS = 32
POOL_SIZE = 16
ACTION_STRIDE = 64  # action float = attack_row + size_fraction + ACTION_STRIDE * (diplomacy_code + 1)
CAN_ATTACK, CAN_REQUEST, ALLIED, REQUESTED_ME = 15, 16, 17, 18  # entity feature columns
DIPLOMACY_PRIOR = 0.02  # reference act rate of the diplomacy gate
DIPLOMACY_PRIOR_PENDING = 0.3  # ... when someone has a request pending to us


def decode_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (attack_row, size_fraction, diplomacy_code); diplomacy_code is -1 for none,
    r for a request to row r, N + r for breaking the alliance with row r (N = slot count)."""
    whole = torch.floor(action).long()
    return whole % ACTION_STRIDE, (action - whole.to(action.dtype)).float(), whole // ACTION_STRIDE - 1


def load_padded(agent: nn.Module, state: dict) -> None:
    """Loads a checkpoint into an agent whose layers may have grown; new weights start at zero."""
    new_state = agent.state_dict()
    for key, old in state.items():
        if key not in new_state:
            continue
        if old.shape == new_state[key].shape:
            new_state[key] = old
        elif old.dim() == new_state[key].dim() and all(n >= o for n, o in zip(new_state[key].shape, old.shape)):
            grown = torch.zeros_like(new_state[key])
            grown[tuple(slice(0, o) for o in old.shape)] = old
            new_state[key] = grown
    agent.load_state_dict(new_state)


def drop_gold_inputs(state: dict) -> dict:
    """Converts a pre-removal checkpoint: deletes the gold column from every layer that read it."""
    pooled = CONV_OUT_CHANNELS * POOL_SIZE * POOL_SIZE
    columns = {
        "backbone.entity_encoder.mlp.0.weight": 5,
        "backbone.merge_mlp.0.weight": pooled,
        "value_head.0.weight": 256,
        "gate_head.weight": 256,
        "diplomacy_gate.weight": 256,
        "size_head.weight": 256,
    }
    state = dict(state)
    for key, col in columns.items():
        if key not in state:
            continue
        w = state[key]
        state[key] = torch.cat([w[:, :col], w[:, col + 1:]], dim=1)
    return state


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
        if "spatial" in ABLATE:
            spatial = torch.zeros_like(spatial)
        pooled_conv = self.pool(self.conv_stack(spatial)).flatten(start_dim=1)
        entity_context, entity_embeds = self.entity_encoder(entities, entity_mask)
        combined = torch.cat([pooled_conv, scalar, entity_context], dim=1)
        return self.merge_mlp(combined), entity_embeds


class Agent(nn.Module):
    """Each tick: a gate (wait or act), a pointer over candidate targets, and a
    categorical attack size. The action is encoded as target_index + size_fraction in one float."""

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
        self.diplomacy_query = nn.Linear(256, 2 * entity_embed_dim)  # request and break pointers
        self.diplomacy_gate = nn.Linear(256 + scalar_dim, 2)
        nn.init.zeros_(self.diplomacy_gate.weight)
        self.break_bias = nn.Parameter(torch.tensor(-2.0))  # breaking starts as a rare choice
        with torch.no_grad():
            self.diplomacy_gate.bias.copy_(torch.log(torch.tensor([1 - DIPLOMACY_PRIOR, DIPLOMACY_PRIOR])))
        self.register_buffer("size_bins", torch.tensor(SIZE_BINS), persistent=False)
        self.size_head = nn.Linear(256 + scalar_dim + entity_embed_dim, len(SIZE_BINS))
        nn.init.zeros_(self.size_head.weight)
        with torch.no_grad():
            self.size_head.bias.copy_(torch.log(torch.tensor(SIZE_INIT_PROBS)))
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

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Samples an action when `action` is None, otherwise scores the given one.
        Returns (action, log_prob, entropy, value)."""
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)
        troop_ratio = scalar[:, 0]

        is_do_nothing = entities[..., 3] > 0.5
        real_target_mask = entity_mask & ~is_do_nothing & (entities[..., CAN_ATTACK] > 0.5)
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

        gate_input = torch.cat([hidden, scalar], dim=-1)
        can_request = entity_mask & (entities[..., CAN_REQUEST] > 0.5)
        can_break = entity_mask & (entities[..., ALLIED] > 0.5)
        request_query, break_query = self.diplomacy_query(hidden).chunk(2, dim=-1)
        scale = self.entity_embed_dim ** 0.5
        diplomacy_scores = torch.cat([
            torch.einsum("bnd,bd->bn", entity_embeds, request_query) / scale,
            torch.einsum("bnd,bd->bn", entity_embeds, break_query) / scale + self.break_bias,
        ], dim=-1).masked_fill(~torch.cat([can_request, can_break], dim=-1), -1e8)
        diplomacy_dist = Categorical(logits=diplomacy_scores)
        can_diplomacy = can_request.any(dim=-1) | can_break.any(dim=-1)
        diplomacy_gate_logits = self.diplomacy_gate(gate_input)
        diplomacy_act_logit = torch.where(can_diplomacy, diplomacy_gate_logits[:, 1], torch.full_like(gate_logits[:, 1], -1e8))
        diplomacy_gate = Categorical(logits=torch.stack([diplomacy_gate_logits[:, 0], diplomacy_act_logit], dim=-1))
        pending = (can_request & (entities[..., REQUESTED_ME] > 0.5)).any(dim=-1)
        prior = torch.where(pending, torch.full_like(troop_ratio, DIPLOMACY_PRIOR_PENDING), torch.full_like(troop_ratio, DIPLOMACY_PRIOR))
        diplomacy_ref = torch.log(torch.stack([1 - prior, prior], dim=-1))
        self.gate_kl = self.gate_kl + (diplomacy_gate.probs * (diplomacy_gate.logits - diplomacy_ref)).sum(dim=-1)

        action_was_sampled = action is None
        if action is None:
            gate_action = gate_dist.sample()
            target_action = target_dist.sample()
            action = torch.where(gate_action == torch.tensor(1), target_action, do_nothing_idx)
            size_dist = self._size_dist(hidden, scalar, entity_embeds, action)
            size_action = self.size_bins[size_dist.sample()]
            size_action = torch.where(action == do_nothing_idx, torch.full_like(size_action, 0.5), size_action)
            diplomacy_code = torch.where(diplomacy_gate.sample() == 1, diplomacy_dist.sample(), torch.full_like(action, -1))
            action_out = action.double() + size_action.double() + ACTION_STRIDE * (diplomacy_code.double() + 1)
        else:
            action_out = action
            target_idx, size_frac, diplomacy_code = decode_action(action)
            size_action = (size_frac.unsqueeze(-1) - self.size_bins).abs().argmin(dim=-1)
            action = target_idx
            size_dist = self._size_dist(hidden, scalar, entity_embeds, action)

        if action_was_sampled:
            size_action = (size_action.unsqueeze(-1) - self.size_bins).abs().argmin(dim=-1)
        acted = (action != do_nothing_idx).long()
        diplomacy_acted = (diplomacy_code >= 0).long()
        log_prob = gate_dist.log_prob(acted) + acted.float() * (
            target_dist.log_prob(action) + size_dist.log_prob(size_action)
        ) + diplomacy_gate.log_prob(diplomacy_acted) + diplomacy_acted.float() * diplomacy_dist.log_prob(diplomacy_code.clamp(min=0))

        # gate entropy is left out; the gate is regularised through gate_kl instead
        num_candidates = real_target_mask.sum(dim=-1).float()
        target_entropy_ceiling = torch.where(num_candidates > 1, torch.log(num_candidates), torch.ones_like(num_candidates))
        normalized_target_entropy = target_dist.entropy() / target_entropy_ceiling
        p_act = gate_dist.probs[:, 1]
        entropy = p_act * normalized_target_entropy + p_act.detach() * size_dist.entropy()

        return action_out, log_prob, entropy, self.value_head(torch.cat([hidden, scalar], dim=-1))

    def _size_dist(self, hidden, scalar, entity_embeds, target_idx):
        emb = entity_embeds[torch.arange(entity_embeds.shape[0], device=entity_embeds.device), target_idx]
        return Categorical(logits=self.size_head(torch.cat([hidden, scalar, emb], dim=-1)))
