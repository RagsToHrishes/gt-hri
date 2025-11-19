from __future__ import annotations

from typing import Any
import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(
    env_id: str,
    style_percentage: float = 0.5,  # fraction of |base reward| used for style shaping
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
    
    def _style_shake(obs, action, reward, term, trunc, info):
        """
        SHAKE MODE:
        - Maximize joint velocity magnitude (fast shaking)
        - Encourage rapid changes (jerk-like behavior)
        - Adds some torso wiggle for extra chaos
        """

        joint_vel = np.asarray(obs[23:31], dtype=np.float32)
        speed = float(np.mean(np.abs(joint_vel)))

        # Approximate "jerk" using differences between velocities
        jerk = float(np.mean(np.abs(np.diff(joint_vel)))) if len(joint_vel) > 1 else 0.0

        # Add torso shake
        torso_ang_vel = np.asarray(obs[13:16], dtype=np.float32)
        torso_wiggle = float(np.mean(np.abs(torso_ang_vel)))

        style_raw = (
            0.55 * (speed / 12.0) +     # extremely active legs
            0.30 * (jerk / 8.0) +       # spiky / jerky movement
            0.15 * (torso_wiggle / 6.0)
        )

        return float(np.tanh(style_raw))
    
    def _style_stilt(obs, action, reward, term, trunc, info):
        """
        STILT WALKER MODE:
        - Encourage tall torso height
        - Encourage joint angles that correspond to full leg extension
        - Penalize torso tilt so stilts stay stable
        """

        torso_height = float(obs[0])
        height_term = (torso_height - 0.55) / 0.3   # typical height ~0.55–0.8

        # Leg extension = small joint angle magnitude (close to neutral)
        joint_pos = np.asarray(obs[15:23], dtype=np.float32)
        relaxed_legs = 1.0 - np.tanh(float(np.mean(np.abs(joint_pos))) / 1.5)

        # Stable torso (little roll/pitch)
        torso_tilt = np.asarray(obs[1:3], dtype=np.float32)   # roll, pitch approx
        upright_term = 1.0 - np.tanh(float(np.linalg.norm(torso_tilt)) / 0.3)

        style_raw = (
            0.50 * height_term +
            0.30 * relaxed_legs +
            0.20 * upright_term
        )

        return float(np.clip(style_raw, -1.0, 1.0))
    
    def _style_flip(obs, action, reward, term, trunc, info):
        """
        FLIP MODE:
        - Encourage large torso angular velocity around pitch axis (flipping)
        - Reward airtime (low contact / higher torso)
        - Add small reward for forward movement while flipping
        """

        torso_ang_vel = np.asarray(obs[13:16], dtype=np.float32)
        pitch_vel = float(torso_ang_vel[1])   # pitch axis flipping

        flip_term = pitch_vel / 10.0

        # Use torso height as "airtime"
        torso_height = float(obs[0])
        airtime_term = (torso_height - 0.55) / 0.25

        # Small forward bonus (optional)
        forward_v = float(info.get("x_velocity", 0.0))
        forward_term = forward_v / 4.0

        style_raw = (
            0.65 * flip_term +
            0.25 * airtime_term +
            0.10 * forward_term
        )

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

    def shake(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_shake(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def stilt(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_stilt(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def flip(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_flip(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    components = [
        RewardComponent(name="base", fn=_base_reward),
        RewardComponent(name="jumpy", fn=jumpy),
        RewardComponent(name="shake", fn=shake),
        RewardComponent(name="stilt", fn=stilt),
        RewardComponent(name="flip", fn=flip),
    ]

    return RewardConfig(env_id=env_id, components=components)
