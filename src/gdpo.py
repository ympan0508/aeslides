"""
Reward-decoupled normalization plugin for Slime.

This module replaces the default post-process function with reward-decoupled
normalization to improve numerical stability and optimization behavior in
multi-reward scenarios.

Usage:
1. Ensure your custom reward function returns a dict containing:
   - each reward component to be processed
   - one total reward
2. Modify the configuration section at the top of this file to specify:
   - which reward components should be normalized
   - their corresponding weights (usually all set to 1.0)
3. Place this file inside the slime package, and in your launch script set:
   --custom-reward-post-process-path=path.to.this.script.post_process_rewards
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

from slime.utils.types import Sample


# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------

# Reward component keys to be normalized independently and then combined.
REWARD_COMPONENT_KEYS = (
    "aspect_ratio_reward",
    "whitespace_reward",
    "overlap_reward",
    "centroid_reward",
)

# Weight for each reward component above.
# It is recommended to keep all weights as 1.0 unless you have a clear reason
# to scale certain reward components differently.
REWARD_COMPONENT_WEIGHTS = {
    "aspect_ratio_reward": 1.0,
    "whitespace_reward": 1.0,
    "overlap_reward": 1.0,
    "centroid_reward": 1.0,
}

# Key for the total/raw reward in sample.reward.
TOTAL_REWARD_KEY = "total_reward"

# Numerical stability epsilon.
EPS = 1e-4


def _group_positions(samples: list[Sample]) -> list[list[int]]:
    grouped: OrderedDict[int, list[int]] = OrderedDict()
    for idx, sample in enumerate(samples):
        group_key = sample.group_index if sample.group_index is not None else idx
        grouped.setdefault(group_key, []).append(idx)
    return list(grouped.values())


def _validate_config() -> None:
    missing_weight_keys = [key for key in REWARD_COMPONENT_KEYS if key not in REWARD_COMPONENT_WEIGHTS]
    if missing_weight_keys:
        raise ValueError(
            "Missing weights for configured reward component keys: "
            f"{missing_weight_keys!r}. "
            "Please ensure every key in REWARD_COMPONENT_KEYS has a corresponding "
            "entry in REWARD_COMPONENT_WEIGHTS."
        )


def _validate_reward_dict(reward_dict: Mapping[str, object], reward_keys: list[str]) -> None:
    missing_keys = [key for key in reward_keys if key not in reward_dict]
    if missing_keys:
        raise ValueError(
            "Reward post process requires the reward dict to contain keys "
            f"{tuple(reward_keys)!r}, but missing {missing_keys!r}."
        )


def _extract_reward_matrix(samples: list[Sample], reward_keys: list[str]) -> torch.Tensor:
    rewards = []
    for sample in samples:
        if not isinstance(sample.reward, Mapping):
            raise TypeError(
                "Reward post process expects each sample.reward to be a dict-like object. "
                f"Got {type(sample.reward)!r} for sample index={sample.index}."
            )
        _validate_reward_dict(sample.reward, reward_keys)
        rewards.append([float(sample.reward[key]) for key in reward_keys])
    return torch.nan_to_num(torch.tensor(rewards, dtype=torch.float32))


def _extract_raw_rewards(
    samples: list[Sample],
    reward_matrix: torch.Tensor | None = None,
    reward_weights: torch.Tensor | None = None,
) -> list[float]:
    raw_rewards = []
    for idx, sample in enumerate(samples):
        if isinstance(sample.reward, Mapping) and TOTAL_REWARD_KEY in sample.reward:
            raw_rewards.append(float(sample.reward[TOTAL_REWARD_KEY]))
        elif reward_matrix is not None and reward_weights is not None:
            raw_rewards.append(float((reward_matrix[idx] * reward_weights).sum().item()))
        else:
            raw_rewards.append(0.0)
    return raw_rewards


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Apply reward-decoupled normalization to user-defined reward components."""

    if not samples:
        return [], []

    _validate_config()

    first_reward = samples[0].reward
    if not isinstance(first_reward, Mapping):
        raise TypeError(
            "Reward post process requires dict rewards. "
            f"Got {type(first_reward)!r} for the first sample."
        )

    reward_keys = list(REWARD_COMPONENT_KEYS)
    if not reward_keys:
        return _extract_raw_rewards(samples), [0.0] * len(samples)

    _validate_reward_dict(first_reward, reward_keys)

    reward_matrix = _extract_reward_matrix(samples, reward_keys)
    reward_weights = torch.tensor(
        [float(REWARD_COMPONENT_WEIGHTS[key]) for key in reward_keys],
        dtype=torch.float32,
    )

    grouped_positions = _group_positions(samples)
    all_reward_advantages = []

    for reward_idx in range(len(reward_keys)):
        reward_values = reward_matrix[:, reward_idx]
        reward_advantage = torch.zeros_like(reward_values)

        for positions in grouped_positions:
            pos_tensor = torch.tensor(positions, dtype=torch.long)
            grouped_values = reward_values.index_select(0, pos_tensor)
            grouped_mean = grouped_values.mean()
            grouped_std = torch.nan_to_num(
                grouped_values.std(dim=0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            reward_advantage.index_copy_(
                0,
                pos_tensor,
                (grouped_values - grouped_mean) / (grouped_std + EPS),
            )

        all_reward_advantages.append(reward_advantage)

    combined_reward_advantage = torch.stack(all_reward_advantages, dim=1)
    pre_batch_norm_advantage = torch.nan_to_num(
        (combined_reward_advantage * reward_weights.unsqueeze(0)).nansum(dim=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    batch_mean = pre_batch_norm_advantage.mean()
    batch_std = torch.nan_to_num(
        pre_batch_norm_advantage.std(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    normalized_advantages = (pre_batch_norm_advantage - batch_mean) / (batch_std + EPS)

    raw_rewards = _extract_raw_rewards(
        samples=samples,
        reward_matrix=reward_matrix,
        reward_weights=reward_weights,
    )

    return raw_rewards, normalized_advantages.tolist()


__all__ = ["post_process_rewards"]
