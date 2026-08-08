import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class EntityEncoder(nn.Module):
    """Shared per-entity MLP + attention pooling (Deep Sets / Set Transformer style).

    Permutation-invariant and works for any entity count N: the same weights embed
    every entity, and a learned query attends over them into one fixed-size context
    vector - no hard-coded slot count baked into what a given position "means".
    """

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
# Collapses the conv output to a fixed 6x6 grid regardless of map size, instead of
# flattening the full H*W grid straight into a Linear layer. The old flatten made that
# one layer ~97-99% of the network's entire param count (worse on bigger maps, since
# it scales with H*W) and tied the network's shape to a specific map resolution.
# Pooling keeps coarse spatial layout (which region is contested) while making the
# backbone's size independent of map dimensions - and, as a side effect, means the
# network no longer needs height/width to be constructed.
POOL_SIZE = 6


class Backbone(nn.Module):
    """Shared CNN + entity encoder + merge MLP: observation -> one hidden vector,
    reused by both the value head and the pointer action head.
    """

    def __init__(self, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
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
    def __init__(self, scalar_dim: int, entity_dim: int, entity_embed_dim: int = 64):
        super().__init__()
        self.entity_embed_dim = entity_embed_dim
        self.backbone = Backbone(scalar_dim, entity_dim, entity_embed_dim)
        self.value_head = nn.Linear(256, 1)
        # Projects backbone hidden state into entity-embedding space so it can be
        # dot-producted against each candidate's embedding (pointer-network style
        # target selection) - replaces the old fixed Linear(256, TOTAL_ACTIONS) head.
        self.query_proj = nn.Linear(256, entity_embed_dim)
        # Separates "should I act at all" from "which target" into its own binary
        # decision (see get_action_and_value) instead of do_nothing being one option
        # competing against however many attack candidates happen to exist that tick -
        # the entangled-softmax problem diagnosed this session (CLAUDE.md).
        self.gate_head = nn.Linear(256, 2)  # logits: [wait, act]

    @classmethod
    def from_observation(cls, spatial: np.ndarray, scalar: np.ndarray, entities: np.ndarray) -> "Agent":
        """Build an Agent sized to match a real observation (e.g. from
        encode_observation / build_entities) instead of hand-passed dimensions -
        every dimension is read directly off real data. spatial's shape no longer
        matters for sizing (pooling makes the backbone map-size-independent) but is
        still accepted to keep call sites symmetric with the other two arrays.
        """
        scalar_dim = scalar.shape[0]
        entity_dim = entities.shape[-1]
        return cls(scalar_dim=scalar_dim, entity_dim=entity_dim)

    def get_value(
        self, spatial: torch.Tensor, scalar: torch.Tensor, entities: torch.Tensor, entity_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden, _ = self.backbone(spatial, scalar, entities, entity_mask)
        return self.value_head(hidden)

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        return_probs: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)

        # column 3 = is_do_nothing (see action_space.py's ENTITY_FEATURES layout).
        # Excluding it from target scoring, and deciding act-vs-wait as its own gate,
        # keeps "should I attack at all" a stable binary choice instead of one option
        # diluted by however many attack candidates happen to exist this tick.
        is_do_nothing = entities[..., 3] > 0.5
        real_target_mask = entity_mask & ~is_do_nothing
        can_act = real_target_mask.any(dim=-1)
        # first (only) True per row - unless do_nothing was truncated away by
        # pad_entities' MAX_ENTITIES cap, same edge case noted in watch.py.
        do_nothing_idx = is_do_nothing.float().argmax(dim=-1)

        query = self.query_proj(hidden)
        target_scores = torch.einsum("bnd,bd->bn", entity_embeds, query) / self.entity_embed_dim ** 0.5
        target_scores = target_scores.masked_fill(~real_target_mask, -1e8)
        target_dist = Categorical(logits=target_scores)

        gate_logits = self.gate_head(hidden)
        wait_logit = gate_logits[:, 0]
        # no legal target this tick (e.g. no border candidates) - force wait.
        act_logit = torch.where(can_act, gate_logits[:, 1], torch.full_like(gate_logits[:, 1], -1e8))
        gate_dist = Categorical(logits=torch.stack([wait_logit, act_logit], dim=-1))

        if action is None:
            gate_action = gate_dist.sample()
            target_action = target_dist.sample()
            action = torch.where(gate_action == 1, target_action, do_nothing_idx)

        # act iff the (given or just-sampled) action isn't the do_nothing row -
        # recovers the gate choice uniformly for both the sampling and PPO-recompute
        # paths instead of branching on whether `action` was passed in.
        acted = (action != do_nothing_idx).long()
        # target_dist.log_prob is only meaningful when acted==1; do_nothing's row is
        # masked to -1e8 in target_scores (finite, not inf) so this can't produce a
        # NaN even when acted==0 zeroes its contribution out.
        log_prob = gate_dist.log_prob(acted) + acted.float() * target_dist.log_prob(action)
        # H(joint) = H(gate) + P(act) * H(target|act) - target only has a second
        # decision to make when the gate chose to act.
        entropy = gate_dist.entropy() + gate_dist.probs[:, 1] * target_dist.entropy()

        result = (action, log_prob, entropy, self.value_head(hidden))
        if return_probs:
            # flattened back to one probability per original entity - p(do_nothing)
            # = p(wait), p(candidate j) = p(act) * p(target=j|act) - so callers (e.g.
            # watch.py's --log-probs) don't need to know the gate exists internally.
            flat_probs = gate_dist.probs[:, 1:2] * target_dist.probs
            flat_probs = torch.where(is_do_nothing, gate_dist.probs[:, 0:1].expand_as(flat_probs), flat_probs)
            result = result + (flat_probs,)
        return result
