from __future__ import annotations

from typing import Any
import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(
    env_id: str,
    style_percentage: float = 1,  # e.g. 0.2 ⇒ style can add/subtract up to ~20% of |base|
) -> RewardConfig:
    if env_id != "HalfCheetah-v4":
        raise ValueError("This config is tailored for HalfCheetah-v4.")

    # --- Helpers -----------------------------------------------------------------

    def _base_reward(_obs, _action, _reward, _term, _trunc, _info: dict[str, Any]):
        # Default HalfCheetah env reward (forward - control cost + survive, etc.)
        return float(_reward)

    def _forward_vel(_obs, _action, _reward, _term, _trunc, info: dict[str, Any]):
        # Try to read from info, fall back to reward_run if present
        return float(info.get("x_velocity", info.get("reward_run", 0.0)))

    def _back_front_torques(action):
        # Action: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        a = np.asarray(action, dtype=np.float32)
        back = a[:3]
        front = a[3:]
        return back, front

    def _apply_style(base: float, style_score: float) -> float:
        """
        Apply a style term that is a fixed fraction of the magnitude of the base reward.
        style_score should be roughly in [-1, 1].
        """
        return base + style_percentage * abs(base) * style_score

    eps = 1e-6

    # --- Style definitions (each ≈ in [-1, 1]) -----------------------------------

    # 1) FAST RUNNER: prefers high positive forward velocity
    def _style_fast_forward(obs, action, reward, term, trunc, info):
        v = _forward_vel(obs, action, reward, term, trunc, info)
        v_ref = 5.0  # typical high speed; tune as needed
        return float(np.tanh(v / v_ref))  # ~-1..1, positive if fast forward

    # 2) SLOW / CRUISING: prefers staying near a target speed
    def _style_slow_cruise(obs, action, reward, term, trunc, info):
        v = _forward_vel(obs, action, reward, term, trunc, info)
        target = 0.5
        # 0 when at target, negative as you deviate. Normalize by range.
        range_scale = 3.0
        return float(-np.tanh(abs(v - target) / range_scale))  # closer to target ⇒ closer to 0

    # 3) JUMPY / HIGH TORSO: prefers higher torso
    # obs[0] ≈ torso height (qpos[1])
    def _style_jumpy(obs, action, reward, term, trunc, info):
        torso_height = float(obs[0])
        baseline = 0.5
        height_range = 0.3  # typical extra height range
        excess = max(torso_height - baseline, 0.0)
        return float(np.tanh(excess / height_range))  # more height ⇒ closer to +1

    # 4) FRONT-LEG PREFERENCE: use more front-leg torque than back
    def _style_front_leg(obs, action, reward, term, trunc, info):
        back, front = _back_front_torques(action)
        front_use = float(np.mean(np.abs(front)))
        back_use = float(np.mean(np.abs(back)))
        total = front_use + back_use + eps
        # +1 if only front, -1 if only back, 0 if equal
        return (front_use - back_use) / total

    # 5) BACK-LEG PREFERENCE: mirror of above
    def _style_back_leg(obs, action, reward, term, trunc, info):
        back, front = _back_front_torques(action)
        front_use = float(np.mean(np.abs(front)))
        back_use = float(np.mean(np.abs(back)))
        total = front_use + back_use + eps
        return (back_use - front_use) / total
    
    def _style_wheel_mode(obs, action, reward, term, trunc, info):
        """
        Wheel mode:
        - Encourage large torso angular velocity (continuous flips)
        - Slight preference for forward movement while flipping
        - Stable spin is better than chaotic wobble
        """

        torso_ang_vel = float(obs[2])  # pitch angular velocity

        # Encourage big rotation magnitude
        spin_term = torso_ang_vel / 15.0     # tuned for cheetah typical limits

        # Encourage forward velocity slightly (not required)
        forward_v = float(info.get("x_velocity", 0.0))
        fwd_term = forward_v / 5.0

        # Blend: mostly spin
        style_raw = 0.8 * spin_term + 0.2 * fwd_term

        return float(np.tanh(style_raw))
    
    def _style_chaos(obs, action, reward, term, trunc, info):
        """
        Chaos mode:
        - Reward large joint velocities
        - Reward rapid oscillation in hip/knee joints
        - Add some torso pitching chaos
        """

        # Joint velocities ~ obs[8:]
        joint_vels = np.asarray(obs[8:], dtype=np.float32)
        joint_speed = float(np.mean(np.abs(joint_vels)))

        # Torso chaos: large pitch velocity
        torso_ang_vel = abs(float(obs[2]))

        # Combine into a chaos score
        chaos_raw = (
            0.75 * (joint_speed / 10.0) +      # big joint movement
            0.25 * (torso_ang_vel / 10.0)      # torso chaos
        )

        return float(np.tanh(chaos_raw))

    # --- Final reward components: base + percentage * style ----------------------

    def fast_forward(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_fast_forward(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def slow_cruise(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_slow_cruise(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def jumpy(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_jumpy(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def front_leg_pref(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_front_leg(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def back_leg_pref(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_back_leg(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def wheel_mode(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_wheel_mode(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def chaos(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_chaos(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    components = [
        RewardComponent(name="base", fn=_base_reward),
        RewardComponent(name="slow_cruise", fn=slow_cruise),
        RewardComponent(name="jumpy", fn=jumpy),
        RewardComponent(name="wheel_mode", fn=wheel_mode),
        RewardComponent(name="chaos", fn=chaos),
    ]

    return RewardConfig(env_id=env_id, components=components)
