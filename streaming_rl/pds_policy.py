"""
pds_policy.py - PDSActorCriticPolicy: SB3's ActorCriticPolicy plus a
second, fully separate critic head Vtilde_psi (the PDS critic), per the
PDS design plan Section 6 (no shared trunk with the actor/ordinary critic
- matches the EHS paper's own choice of fully separate architectures).

pds_value_net operates directly on the raw 68-dim PDS observation
(streaming_rl.pds.PDS_OBS_DIM) - no features_extractor/mlp_extractor
needed, since that machinery is built around the 4-dim ordinary
observation space and the PDS vector's own normalization is already
handled upstream (VecNormalize on the 4 continuous components via
info["terminal_observation"], raw 0/1 tile mask - PDS design plan,
Section 9).
"""

from functools import partial

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from streaming_rl.pds import PDS_OBS_DIM


class PDSActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Schedule,
                 pds_obs_dim: int = PDS_OBS_DIM, pds_net_arch: list = None,
                 *args, **kwargs):
        # Set BEFORE super().__init__(), which calls self._build(lr_schedule)
        # internally (ActorCriticPolicy.__init__'s own last line) - _build()
        # below needs these two already present when it runs.
        self.pds_obs_dim = int(pds_obs_dim)
        self.pds_net_arch = list(pds_net_arch) if pds_net_arch is not None else [64, 64]
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_pds_value_net(self) -> nn.Module:
        layers = []
        prev_dim = self.pds_obs_dim
        for hidden_dim in self.pds_net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _build(self, lr_schedule: Schedule) -> None:
        # Let the base class build everything it normally does first (actor,
        # ordinary critic, mlp_extractor, and - at the very end - its own
        # self.optimizer over self.parameters()). Add the PDS critic head
        # afterward and REBUILD the optimizer so it also covers
        # pds_value_net's parameters. Rebuilding rather than duplicating the
        # base class's action-distribution branching here keeps this robust
        # to future SB3 changes inside _build() itself.
        super()._build(lr_schedule)

        self.pds_value_net = self._build_pds_value_net()
        if self.ortho_init:
            self.pds_value_net.apply(partial(self.init_weights, gain=1))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def predict_pds_values(self, pds_obs: th.Tensor) -> th.Tensor:
        """Vtilde_psi(omega~^t) - the PDS critic's own forward pass, no
        feature extraction (see module docstring)."""
        return self.pds_value_net(pds_obs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(pds_obs_dim=self.pds_obs_dim, pds_net_arch=self.pds_net_arch))
        return data
