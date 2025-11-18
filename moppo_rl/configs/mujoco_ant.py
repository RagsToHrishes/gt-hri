from __future__ import annotations

from typing import Any
import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(
    env_id: str,
    style_percentage: float = 0.2,  # fraction of |base reward| used for style shaping
) -> RewardConfig:
    """
    Multi-objective reward config for Ant-v4.

    Each objective:
        R_i = base_reward + style_percentage * |base_reward| * style_score_i

    where style_score_i is in [-1, 1] and encodes a preference for a particular gait.
    """
    if env_id != "Ant-v4":
        raise ValueError("This config is tailored for Ant-v4.")

    eps = 1e-6

    # --- Helpers -----------------------------------------------------------------

    def _base_reward(
        _obs, _action, _reward, _term, _trunc, _info: dict[str, Any]
    ) -> float:
        # Default Ant reward (forward + healthy - ctrl - contact, etc.)
        return float(_reward)

    def _apply_style(base: float, style_score: float) -> float:
        """
        Apply a style term that is a fixed fraction of the magnitude of the base reward.
        style_score should be roughly in [-1, 1].
        """
        return base + style_percentage * abs(base) * float(
            np.clip(style_score, -1.0, 1.0)
        )

    # Observation convention for Ant (exclude current x,y positions):
    # obs[0] ~ torso height (z). We only rely on that here.

    # Ant actions: 8 joints, roughly:
    # [front_left_hip, front_left_knee,
    #  front_right_hip, front_right_knee,
    #  back_left_hip,  back_left_knee,
    #  back_right_hip, back_right_knee]

    def _split_legs(action):
        a = np.asarray(action, dtype=np.float32)
        # Front vs back
        front = a[0:4]
        back = a[4:8]
        # Left vs right
        left = a[[0, 1, 4, 5]]
        right = a[[2, 3, 6, 7]]
        return front, back, left, right

    # --- Style terms (each returns ~[-1, 1]) -------------------------------------

    # 1) JUMPY: prefers higher torso (bouncier ant)
    def _style_jumpy(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])
        baseline_height = 0.6   # rough standing height; tune as needed
        height_range = 0.2      # how much extra height counts as “high”

        excess = max(torso_z - baseline_height, 0.0)
        style_raw = excess / max(height_range, eps)
        return float(np.tanh(style_raw))

    # 2) LOW-SLUNG: prefers lower COM (crouched / crab-like gait)
    def _style_low_slung(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])
        baseline_height = 0.6
        height_range = 0.2

        style_raw = (baseline_height - torso_z) / max(height_range, eps)
        return float(np.tanh(style_raw))

    # 3) FRONT-LEG PREFERENCE: uses front legs more than back
    def _style_front_leg_pref(obs, action, reward, term, trunc, info):
        front, back, left, right = _split_legs(action)
        front_use = float(np.mean(np.abs(front)))
        back_use = float(np.mean(np.abs(back)))
        total = front_use + back_use + eps
        # +1 if only front, -1 if only back, 0 if equal
        return (front_use - back_use) / total

    # 4) LEFT-LEG PREFERENCE: uses left legs more than right
    def _style_left_leg_pref(obs, action, reward, term, trunc, info):
        front, back, left, right = _split_legs(action)
        left_use = float(np.mean(np.abs(left)))
        right_use = float(np.mean(np.abs(right)))
        total = left_use + right_use + eps
        return (left_use - right_use) / total

    # 5) ENERGY EFFICIENT: prefers smaller torques overall
    def _style_energy_efficient(obs, action, reward, term, trunc, info):
        a = np.asarray(action, dtype=np.float32)
        ctrl_norm = float(np.mean(a * a))  # mean squared torque

        ref_ctrl = 0.1  # reference control magnitude; tune if needed
        style_raw = (ref_ctrl - ctrl_norm) / max(ref_ctrl, eps)
        return float(np.tanh(style_raw))

    # --- Final reward components: base + style -----------------------------------

    def jumpy(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_jumpy(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def low_slung(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_low_slung(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def front_leg_pref(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_front_leg_pref(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def left_leg_pref(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_left_leg_pref(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def energy_efficient(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_energy_efficient(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    components = [
        RewardComponent(name="base", fn=_base_reward),
        RewardComponent(name="jumpy", fn=jumpy),
        RewardComponent(name="low_slung", fn=low_slung),
        RewardComponent(name="front_leg_pref", fn=front_leg_pref),
        RewardComponent(name="left_leg_pref", fn=left_leg_pref),
        RewardComponent(name="energy_efficient", fn=energy_efficient),
    ]

    return RewardConfig(env_id=env_id, components=components)
