import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


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


class Agent(nn.Module):
    def __init__(self, spatial_channels: int, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.entity_embed_dim = entity_embed_dim
        self.backbone = Backbone(spatial_channels, scalar_dim, entity_dim, entity_embed_dim)
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
        hidden, _ = self.backbone(spatial, scalar, entities, entity_mask)
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
        can_act = real_target_mask.any(dim=-1)
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

        if action is None:
            gate_action = gate_dist.sample()
            target_action = target_dist.sample()
            action = torch.where(gate_action == torch.tensor(1), target_action, do_nothing_idx)

        # acted iff action isn't the do_nothing row - recovers the gate choice uniformly
        acted = (action != do_nothing_idx).long()
        log_prob = gate_dist.log_prob(acted) + acted.float() * target_dist.log_prob(action)
        # This is NOT H(joint) = H(gate) + P(act)*H(target|act) (the true joint entropy,
        # which the log_prob above already correctly reflects) - it's only used as train.py's
        # PPO exploration bonus, and the true-entropy formula is wrong for that purpose.
        # d[P(act)*H(target)]/dP(act) = H(target) >= 0 unconditionally, so weighting by
        # P(act) makes "raise P(act)" a free way to harvest more bonus whenever there are
        # multiple live targets (H(target) > 0) - i.e. almost every tick bordering an
        # opponent - regardless of whether acting is actually good. That's a standing
        # gradient pressure toward attacking, fighting directly against the patience this
        # gate exists to learn. Summing unweighted removes that coupling: each term is
        # rewarded for its own uncertainty, independent of the other's probability.
        #
        # target_dist.entropy() is normalized by its own ceiling (log(num_candidates)) -
        # a fixed-checkpoint nations=1/2/3 sweep (see CLAUDE.md) confirmed its RAW
        # magnitude grows ~9.7x from nations=1->3 while candidate count only grows ~2.8x,
        # and target_entropy/ceiling stays pinned near 0.9-0.99 at every nations count -
        # i.e. the distribution is consistently ~uniform over however many candidates
        # exist, not "more confused." Unnormalized, the same ENTROPY_COEF applies an
        # ever-growing structural push as the curriculum scales to more opponents,
        # independent of any real change in target-selection confidence. Dividing by the
        # ceiling makes this term scale-invariant to opponent count instead.
        num_candidates = real_target_mask.sum(dim=-1).float()
        target_entropy_ceiling = torch.where(num_candidates > 1, torch.log(num_candidates), torch.ones_like(num_candidates))
        normalized_target_entropy = target_dist.entropy() / target_entropy_ceiling
        entropy = gate_dist.entropy() + normalized_target_entropy

        result = (action, log_prob, entropy, self.value_head(torch.cat([hidden, scalar], dim=-1)))
        if return_probs:
            # p(do_nothing)=p(wait), p(candidate j)=p(act)*p(target=j|act)
            flat_probs = gate_dist.probs[:, 1:2] * target_dist.probs
            flat_probs = torch.where(is_do_nothing, gate_dist.probs[:, 0:1].expand_as(flat_probs), flat_probs)
            result = result + (flat_probs,)
        if return_entropy_components:
            # disentangles "gate is less confident" (behavioral) from "target_dist has
            # more candidates to spread over" (structural - grows with nations count,
            # independent of gate confidence) - see CLAUDE.md's entropy-scaling hypothesis.
            # num_candidates lets a caller normalize target_entropy against its own
            # ceiling (log(num_candidates)) instead of comparing raw values across
            # different opponent counts. Raw (unnormalized) target_dist.entropy() is
            # returned here regardless of the normalization applied to `entropy` above,
            # so this diagnostic keeps measuring the same raw quantity either way.
            result = result + (gate_dist.entropy(), target_dist.entropy(), num_candidates)
        return result
