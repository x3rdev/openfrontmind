import torch
import torch.nn as nn
from torch.distributions import Categorical


class Backbone(nn.Module):
    """Shared CNN + merge MLP: observation -> one hidden vector, reused by
    both the value and policy heads instead of each learning its own
    understanding of the board from scratch.
    """

    def __init__(self, height: int, width: int, scalar_dim: int):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            flattened_size = self.conv_stack(dummy).flatten(start_dim=1).shape[1]

        self.merge_mlp = nn.Sequential(
            nn.Linear(flattened_size + scalar_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

    def forward(self, spatial: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        flattened = self.conv_stack(spatial).flatten(start_dim=1)
        combined = torch.cat([flattened, scalar], dim=1)
        return self.merge_mlp(combined)


class Agent(nn.Module):
    def __init__(self, height: int, width: int, scalar_dim: int, total_actions: int):
        super().__init__()
        self.backbone = Backbone(height, width, scalar_dim)
        self.value_head = nn.Linear(256, 1)
        self.policy_head = nn.Linear(256, total_actions)

    def get_value(self, spatial: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(spatial, scalar))

    def get_action_and_value(
        self,
        spatial: torch.Tensor,
        scalar: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(spatial, scalar)
        logits = self.policy_head(hidden).masked_fill(~mask, -1e8)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value_head(hidden)
