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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, entity_embeds = self.backbone(spatial, scalar, entities, entity_mask)
        query = self.query_proj(hidden)
        scores = torch.einsum("bnd,bd->bn", entity_embeds, query) / self.entity_embed_dim ** 0.5
        scores = scores.masked_fill(~entity_mask, -1e8)
        dist = Categorical(logits=scores)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value_head(hidden)
