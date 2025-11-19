from __future__ import annotations

from typing import Any
import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(
    env_id: str,
    style_percentage: float = 1,  # fraction of |base reward| used for style shaping
) -> RewardConfig:
    """
    Multi-objective reward config for Hopper-v4.

    Each objective:
        R_i = base_reward + style_percentage * |base_reward| * style_score_i

    where style_score_i is in [-1, 1] and encodes a preference for a particular gait.
    """
    if env_id != "Hopper-v4":
        raise ValueError("This config is tailored for Hopper-v4.")

    eps = 1e-6

    # --- Helpers -----------------------------------------------------------------

    def _base_reward(
        _obs, _action, _reward, _term, _trunc, _info: dict[str, Any]
    ) -> float:
        # Default Hopper reward from the env (forward + healthy - ctrl, etc.)
        return float(_reward)

    def _apply_style(base: float, style_score: float) -> float:
        """
        Apply a style term that is a fixed fraction of the magnitude of the base reward.
        style_score should be roughly in [-1, 1].
        """
        return base + style_percentage * abs(base) * float(np.clip(style_score, -1.0, 1.0))

    def _forward_vel(
        _obs, _action, _reward, _term, _trunc, info: dict[str, Any]
    ) -> float:
        # Hopper info usually has x_velocity / reward_forward.
        return float(info.get("x_velocity", info.get("reward_forward", 0.0)))

    # Observation (Hopper-v4, exclude_current_positions_from_observation=True):
    # obs[0] ≈ torso height (z)
    # obs[1] ≈ torso pitch angle (0 = upright, +/− = leaning)
    # the rest: joint positions/velocities

    # --- Style terms (each returns ~[-1, 1]) -------------------------------------

    # 1) JUMPY: prefers higher COM (bigger hops)
    def _style_jumpy(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])
        baseline_height = 1.25  # rough standing height, tune if needed
        height_range = 0.3      # how much extra height counts as “big hop”

        excess = max(torso_z - baseline_height, 0.0)
        style_raw = excess / max(height_range, eps)
        return float(np.tanh(style_raw))

    # 2) UPRIGHT: prefers small torso pitch magnitude (good balance)
    def _style_upright(obs, action, reward, term, trunc, info):
        torso_angle = float(obs[1])  # 0 = upright

        angle_scale = 0.4  # ~ where tilt is “significant”
        # 1 when angle ≈ 0, tends towards -1 when |angle| is large
        style_raw = 1.0 - 2.0 * np.tanh(abs(torso_angle) / max(angle_scale, eps))
        return float(np.clip(style_raw, -1.0, 1.0))

    # 3) CROUCHED: prefers lower COM (bent leg / crouched hopping)
    def _style_crouch(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])
        baseline_height = 1.25
        height_range = 0.3

        style_raw = (baseline_height - torso_z) / max(height_range, eps)
        return float(np.tanh(style_raw))

    # 4) FAST FORWARD: prefers higher forward velocity
    def _style_fast_forward(obs, action, reward, term, trunc, info):
        v = _forward_vel(obs, action, reward, term, trunc, info)
        v_ref = 3.0  # typical high speed; tune as needed
        style_raw = v / max(v_ref, eps)
        return float(np.tanh(style_raw))

    # 5) ENERGY EFFICIENT: prefers smaller torques (less control effort)
    def _style_energy_efficient(obs, action, reward, term, trunc, info):
        a = np.asarray(action, dtype=np.float32)
        ctrl_norm = float(np.mean(a * a))  # mean squared torque

        ref_ctrl = 0.05  # reference control magnitude; tune if needed
        style_raw = (ref_ctrl - ctrl_norm) / max(ref_ctrl, eps)
        return float(np.tanh(style_raw))
    
    def _style_wobble(obs, action, reward, term, trunc, info):
        torso_pitch = float(obs[1])
        torso_pitch_vel = float(obs[2])

        wobble_mag = abs(torso_pitch_vel) / 5.0
        off_center = abs(torso_pitch) / 0.4

        style_raw = 0.7 * wobble_mag + 0.3 * off_center
        return float(np.tanh(style_raw))
        

    # --- Final reward components: base + style -----------------------------------

    def jumpy(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_jumpy(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def upright(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_upright(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def crouch(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_crouch(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def fast_forward(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_fast_forward(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def energy_efficient(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_energy_efficient(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)
    
    def wobble(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_wobble(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    components = [
        RewardComponent(name="base", fn=_base_reward),
        RewardComponent(name="jumpy", fn=jumpy),
        RewardComponent(name="upright", fn=upright),
        RewardComponent(name="crouch", fn=crouch),
        RewardComponent(name="fast_forward", fn=fast_forward),
        RewardComponent(name="wobble", fn=wobble),
    ]
        
    return RewardConfig(env_id=env_id, components=components)
