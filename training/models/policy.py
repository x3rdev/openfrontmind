import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical


SIZE_MIN, SIZE_MAX = 0.01, 0.999


class EntityEncoder(nn.Module):
    """Shared per-entity MLP + attention pooling - permutation-invariant, any N."""

    def __init__(self, entity_dim: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.pool_query = nn.Parameter(torch.randn(embed_dim) * embed_dim ** -0.5)

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """entities: (B, N, entity_dim), mask: (B, N) bool, True = real entity.
        Returns (pooled context (B, embed_dim), per-entity embeddings (B, N, embed_dim)).
        """
        embeds = self.mlp(entities)
        scores = (embeds @ self.pool_query) / self.embed_dim ** 0.5
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * embeds).sum(dim=1)
        return pooled, embeds


CONV_OUT_CHANNELS = 32
# fixed-size regardless of map dims, unlike flattening the full H*W grid (old approach,
# ~97-99% of all params in one layer). Backbone no longer needs height/width either.
POOL_SIZE = 16


class Backbone(nn.Module):
    """Shared CNN + entity encoder + merge MLP -> one hidden vector for both heads."""

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


# threshold grid for the decision-level action: attack fires once troop_ratio >= tau
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)


class Agent(nn.Module):
    def __init__(self, spatial_channels: int, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.entity_embed_dim = entity_embed_dim
        self.backbone = Backbone(spatial_channels, scalar_dim, entity_dim, entity_embed_dim)
        # critic gets its own trunk so policy gradients can't move the value representation
        self.separate_critic = os.environ.get("OFM_SEPARATE_CRITIC", "0") == "1"
        self.detach_gate_entropy = os.environ.get("OFM_DETACH_GATE_ENTROPY", "0") == "1"
        if self.separate_critic:
            self.value_backbone = Backbone(spatial_channels, scalar_dim, entity_dim, entity_embed_dim)
        # more capacity than a bare Linear - see CLAUDE.md, value-trace diagnostic.
        self.value_head = nn.Sequential(
            nn.Linear(256 + scalar_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        # pointer-network target selection: dot-product against each candidate's embedding
        self.query_proj = nn.Linear(256, entity_embed_dim)
        # separates "act at all" from "which target" - see CLAUDE.md's entangled-softmax fix.
        # scalar concatenated directly (same shortcut value_head has above) - troop_ratio is
        # the decisive signal for this decision and shouldn't have to survive compression
        # through the shared backbone bottleneck to reach it, same reasoning that motivated
        # value_head's own scalar skip-connection.
        self.gate_head = nn.Linear(256 + scalar_dim, 2)  # logits: [wait, act]
        self.decision = os.environ.get("OFM_DECISION", "0") == "1"
        self.reserve_floor = float(os.environ.get("OFM_RESERVE_FLOOR", "0"))
        self.gate_kl_mode = float(os.environ.get("OFM_GATE_KL", "0")) > 0
        self.gate_kl = None
        self.use_size = os.environ.get("OFM_SIZE", "0") == "1" and not self.decision
        if self.use_size:
            # attack fraction ~ Beta(1+softplus(a), 1+softplus(b)); action is encoded target_idx + fraction
            self.size_head = nn.Linear(256 + scalar_dim + entity_embed_dim, 2)
            nn.init.zeros_(self.size_head.weight)
            with torch.no_grad():
                self.size_head.bias.copy_(torch.tensor([0.5413, 6.0025]))  # Beta(2, 7), mean ~0.22
        if self.decision:
            self.tau_head = nn.Linear(256 + scalar_dim + entity_embed_dim, len(TAUS))
            nn.init.zeros_(self.tau_head.weight)
            with torch.no_grad():
                self.tau_head.bias.copy_(torch.tensor([-1.0, 0.0, 1.0, 0.0, -1.0]))

    @classmethod
    def from_observation(cls, spatial: np.ndarray, scalar: np.ndarray, entities: np.ndarray) -> "Agent":
        """Sizes an Agent from a real observation instead of hand-passed dimensions."""
        spatial_channels = spatial.shape[0]
        scalar_dim = scalar.shape[0]
        entity_dim = entities.shape[-1]
        return cls(spatial_channels=spatial_channels, scalar_dim=scalar_dim, entity_dim=entity_dim)

    def get_value(
        self, spatial: torch.Tensor, scalar: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        trunk = self.value_backbone if self.separate_critic else self.backbone
        hidden, _ = trunk(spatial, scalar, entities, entity_mask)
        return self.value_head(torch.cat([hidden, scalar], dim=-1))

    def soft_update_from(self, source: "Agent", tau: float) -> None:
        """Polyak/EMA update toward `source`'s current weights - see CLAUDE.md's
        target-network fix for the value-bootstrap oscillation. `tau` is the fraction
        pulled toward source each call (small, e.g. 0.005) so this network moves much
        slower than `source`'s own gradient descent."""
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
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)

        is_do_nothing = entities[..., 3] > 0.5
        is_already_attacking = entities[..., 8] > 0.5
        real_target_mask = entity_mask & ~is_do_nothing & ~is_already_attacking
        can_act = real_target_mask.any(dim=-1) & (scalar[:, 1] >= self.reserve_floor)
        do_nothing_idx = is_do_nothing.float().argmax(dim=-1)  # first True, per watch.py's truncation caveat

        query = self.query_proj(hidden)
        target_scores = torch.einsum("bnd,bd->bn", entity_embeds, query) / self.entity_embed_dim ** 0.5
        target_scores = target_scores.masked_fill(~real_target_mask, -1e8)
        target_dist = Categorical(logits=target_scores)

        gate_logits = self.gate_head(torch.cat([hidden, scalar], dim=-1))
        wait_logit = gate_logits[:, 0]
        # no legal target this tick (e.g. no border candidates) - force wait.
        act_logit = torch.where(can_act, gate_logits[:, 1], torch.full_like(gate_logits[:, 1], -1e8))
        gate_dist = Categorical(logits=torch.stack([wait_logit, act_logit], dim=-1))

        if self.gate_kl_mode:
            # patient reference gate: near-never act below ~0.45 troop ratio, act freely above
            ref_act = 0.005 + 0.495 * torch.sigmoid((scalar[:, 1] - 0.5) / 0.03)
            ref_logp = torch.log(torch.stack([1 - ref_act, ref_act], dim=-1))
            self.gate_kl = (gate_dist.probs * (gate_dist.logits - ref_logp)).sum(dim=-1)
        size_dist = None
        action_out = None
        if action is None:
            gate_action = gate_dist.sample()
            target_action = target_dist.sample()
            action = torch.where(gate_action == torch.tensor(1), target_action, do_nothing_idx)
            if self.use_size:
                size_dist = self._size_dist(hidden, scalar, entity_embeds, action)
                size_action = size_dist.sample().clamp(SIZE_MIN, SIZE_MAX)
                size_action = torch.where(action == do_nothing_idx, torch.full_like(size_action, 0.5), size_action)
                action_out = action.double() + size_action.double()
        elif self.use_size:
            action_out = action
            target_idx = torch.floor(action).long()
            size_action = (action - target_idx.to(action.dtype)).float().clamp(SIZE_MIN, SIZE_MAX)
            action = target_idx
            size_dist = self._size_dist(hidden, scalar, entity_embeds, action)
        if action_out is None:
            action_out = action

        # acted iff action isn't the do_nothing row - recovers the gate choice uniformly
        acted = (action != do_nothing_idx).long()
        log_prob = gate_dist.log_prob(acted) + acted.float() * target_dist.log_prob(action)
        if size_dist is not None:
            log_prob = log_prob + acted.float() * size_dist.log_prob(size_action)

        num_candidates = real_target_mask.sum(dim=-1).float()
        target_entropy_ceiling = torch.where(num_candidates > 1, torch.log(num_candidates), torch.ones_like(num_candidates))
        normalized_target_entropy = target_dist.entropy() / target_entropy_ceiling
        if self.detach_gate_entropy:
            det_logits = self.gate_head(torch.cat([hidden.detach(), scalar], dim=-1))
            det_act = torch.where(can_act, det_logits[:, 1], torch.full_like(det_logits[:, 1], -1e8))
            det_dist = Categorical(logits=torch.stack([det_logits[:, 0], det_act], dim=-1))
            entropy = det_dist.entropy() + det_dist.probs[:, 1].detach() * normalized_target_entropy
        else:
            gate_ent = torch.zeros_like(normalized_target_entropy) if self.gate_kl_mode else gate_dist.entropy()
            entropy = gate_ent + gate_dist.probs[:, 1] * normalized_target_entropy
        if size_dist is not None:
            entropy = entropy + gate_dist.probs[:, 1].detach() * size_dist.entropy()

        # 2026-10-09: tried detaching hidden before value_head (testing whether a
        # shared trunk causes representation collapse under sparse reward, per
        # arXiv 2405.00662) - reverted. Real run showed value_loss climbing
        # monotonically 5x+ over 100 updates with no recovery, unlike every other
        # run's oscillating-but-stable value_loss - cutting the value gradient out of
        # the backbone starved the value head of a representation shaped for its own
        # task, a worse failure than the one being tested for. Collapse (0% win rate,
        # updates 56-71) still happened anyway, so this wasn't masking a real fix.
        value_hidden = self.value_backbone(spatial, scalar, entities, entity_mask)[0] if self.separate_critic else hidden
        result = (action_out, log_prob, entropy, self.value_head(torch.cat([value_hidden, scalar], dim=-1)))
        if return_probs:
            # p(do_nothing)=p(wait), p(candidate j)=p(act)*p(target=j|act)
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

    def _decision_heads(self, spatial, scalar, entities, entity_mask):
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)
        real = entity_mask & ~(entities[..., 3] > 0.5) & ~(entities[..., 8] > 0.5)
        query = self.query_proj(hidden)
        scores = torch.einsum("bnd,bd->bn", entity_embeds, query) / self.entity_embed_dim ** 0.5
        target_dist = Categorical(logits=scores.masked_fill(~real, -1e8))
        # tau sees the probability-weighted embedding of the likely targets
        tgt_ctx = (target_dist.probs.detach().unsqueeze(-1) * entity_embeds).sum(dim=1)
        tau_dist = Categorical(logits=self.tau_head(torch.cat([hidden, scalar, tgt_ctx], dim=-1)))
        value_hidden = self.value_backbone(spatial, scalar, entities, entity_mask)[0] if self.separate_critic else hidden
        value = self.value_head(torch.cat([value_hidden, scalar], dim=-1))
        return tau_dist, target_dist, real, value

    def act_decision(self, spatial, scalar, entities, entity_mask):
        tau_dist, target_dist, real, value = self._decision_heads(spatial, scalar, entities, entity_mask)
        tau_a = tau_dist.sample()
        target_a = target_dist.sample()
        return tau_a, tau_dist.log_prob(tau_a), target_a, target_dist.log_prob(target_a), value, real.any(dim=-1)

    def evaluate_decision(self, spatial, scalar, entities, entity_mask, dec_type, action):
        tau_dist, target_dist, real, value = self._decision_heads(spatial, scalar, entities, entity_mask)
        is_tau = dec_type == 1
        tau_a = action.clamp(0, len(TAUS) - 1)
        target_a = action.clamp(0, entity_mask.shape[1] - 1)
        log_prob = torch.where(is_tau, tau_dist.log_prob(tau_a), target_dist.log_prob(target_a))
        n = real.sum(dim=-1).float()
        ceiling = torch.where(n > 1, torch.log(n), torch.ones_like(n))
        entropy = torch.where(is_tau, tau_dist.entropy(), target_dist.entropy() / ceiling)
        return log_prob, entropy, value
