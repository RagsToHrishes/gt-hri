from __future__ import annotations

from typing import Any
import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(
    env_id: str,
    style_percentage: float = 0.5,  # fraction of |base reward| used for style shaping
) -> RewardConfig:
    """
    Multi-objective reward config for Humanoid-v4.

    Each objective:
        R_i = base_reward + style_percentage * |base_reward| * style_score_i

    where style_score_i is in [-1, 1] and encodes a preference for a particular gait.
    """
    if env_id != "Humanoid-v4":
        raise ValueError("This config is tailored for Humanoid-v4.")

    eps = 1e-6

    # --- Helpers -----------------------------------------------------------------

    def _base_reward(
        _obs, _action, _reward, _term, _trunc, _info: dict[str, Any]
    ) -> float:
        # Default Humanoid reward (forward + healthy - ctrl - contact)
        return float(_reward)

    def _apply_style(base: float, style_score: float) -> float:
        """
        Apply a style term that is a fixed fraction of the magnitude of the base reward.
        style_score should be roughly in [-1, 1].
        """
        return base + style_percentage * abs(base) * float(np.clip(style_score, -1.0, 1.0))

    # Observation conventions (Humanoid-v4 / Humanoid-v5, with exclude_current_positions=True):
    # obs[0]  = torso height (z)
    # obs[1:5] = torso orientation as quaternion (w, x, y, z)
    # obs[22] = x-velocity of torso (forward)
    # obs[24] = z-velocity of torso (vertical)  :contentReference[oaicite:0]{index=0}

    # --- Style terms (each returns ~[-1, 1]) -------------------------------------

    # 1) JUMPY / BOUNCY: prefers higher COM and upward motion
    def _style_jumpy(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])
        torso_z_vel = float(obs[24])

        # Baseline standing height and rough ranges – tune if needed
        baseline_height = 1.2
        height_range = 0.4
        vel_scale = 2.0

        height_term = (torso_z - baseline_height) / max(height_range, eps)
        vel_term = torso_z_vel / max(vel_scale, eps)

        style_raw = height_term + 0.5 * vel_term
        return float(np.tanh(style_raw))

    # 2) UPRIGHT / STABLE: prefers upright torso (low pitch/roll tilt)
    def _style_upright(obs, action, reward, term, trunc, info):
        # Torso orientation quaternion: (w, x, y, z)
        w, x, y, z = map(float, obs[1:5])

        # Treat pitch/roll tilt magnitude as sqrt(x^2 + y^2),
        # ignore yaw (z) so turning is allowed.
        tilt_sq = x * x + y * y
        tilt_scale = 0.2  # tune; ~tilt^2 where it starts to be "bad"

        # 1 when perfectly upright, tends towards -1 for large tilt.
        style_raw = 1.0 - 2.0 * np.tanh(tilt_sq / max(tilt_scale, eps))
        return float(np.clip(style_raw, -1.0, 1.0))

    # 3) CROUCHED GAIT: prefers lower COM (bent knees / crouch)
    def _style_crouch(obs, action, reward, term, trunc, info):
        torso_z = float(obs[0])

        baseline_height = 1.2
        height_range = 0.4

        # Positive if lower than baseline, negative if taller.
        style_raw = (baseline_height - torso_z) / max(height_range, eps)
        return float(np.tanh(style_raw))

    # 4) ARM SWING: prefers more arm motion (swingy arms)
    def _style_arm_swing(obs, action, reward, term, trunc, info):
        # Angular velocities for arm joints in obs indices [39:45]
        # right_shoulder1, right_shoulder2, right_elbow,
        # left_shoulder1, left_shoulder2, left_elbow :contentReference[oaicite:1]{index=1}
        arm_ang_vel = np.asarray(obs[39:45], dtype=np.float32)
        arm_speed = float(np.mean(np.abs(arm_ang_vel)))

        # Reference speed where style ~ 0
        ref_speed = 1.0
        style_raw = (arm_speed - ref_speed) / max(ref_speed, eps)
        return float(np.tanh(style_raw))

    # 5) ENERGY EFFICIENT: prefers using less torque (smaller actions)
    def _style_energy_efficient(obs, action, reward, term, trunc, info):
        a = np.asarray(action, dtype=np.float32)
        ctrl_norm = float(np.mean(a * a))  # mean squared torque

        # Reference control magnitude; smaller than this => positive style
        ref_ctrl = 0.08  # tune if needed
        style_raw = (ref_ctrl - ctrl_norm) / max(ref_ctrl, eps)
        return float(np.tanh(style_raw))
    
       # 5) CRAWLER: prefers limbs close to the floor (promotes crawling gait)
    def _style_crawler(obs, action, reward, term, trunc, info):
        """
        Encourages crawling by rewarding low limb height.
        We approximate limb height by joint angles and torso height:
        - very low torso height suggests crouched/crawling pose
        - bent joints (large angles) indicate crawling posture
        """

        # Torso height (obs[0])
        torso_z = float(obs[0])

        # Joint angles roughly correspond to crouch:
        # Humanoid-v4 joint angles in obs indices (from Gym docs)
        # hip, knee, ankle, shoulder, elbow, etc.
        # We'll focus on lower body + elbows:
        #   legs:   obs[5:11]
        #   arms:   obs[17:22]
        lower_body = np.asarray(obs[5:11], dtype=np.float32)
        arms = np.asarray(obs[17:22], dtype=np.float32)

        # Large magnitudes = bent joints = crawling
        bend_score = float(
            np.mean(np.abs(lower_body)) * 0.7 +
            np.mean(np.abs(arms)) * 0.3
        )

        # Torso closer to ground = more crawling
        baseline_height = 1.2
        crawl_height_term = (baseline_height - torso_z) / 0.5  # ~[-1,1]

        # Combine: bent joints + low torso = crawling
        style_raw = 0.6 * crawl_height_term + 0.4 * bend_score

        return float(np.tanh(style_raw))
    
    # BEYBLADE: encourages spinning rapidly around the vertical axis
    def _style_beyblade(obs, action, reward, term, trunc, info):
        """
        Beyblade mode:
        - Reward yaw angular velocity (spinning)
        - Reward near-upright torso so it can spin stably
        - Slightly reward forward velocity during spin
        """

        # Humanoid angular velocities live in obs indices ~[20:23]
        # According to Gym: these are (angular_velocity_x, y, z)
        # yaw spin = angular_velocity_z
        ang_vel = np.asarray(obs[20:23], dtype=np.float32)
        yaw_spin = float(ang_vel[2])     # spinning around vertical axis

        # Encourage BIG spinning
        spin_term = yaw_spin / 8.0      # normalize scale, tune if needed

        # Encourage upright posture (pitch/roll small)
        # obs[1:5] = quaternion (w, x, y, z)
        w, x, y, z = map(float, obs[1:5])
        tilt = np.sqrt(x*x + y*y)       # tilt magnitude
        upright_term = 1.0 - np.tanh(tilt / 0.3)

        # Encourage forward movement while spinning (optional, small weight)
        forward_v = float(info.get("x_velocity", 0.0))
        forward_term = forward_v / 4.0

        # Combine behaviors
        style_raw = (
            0.65 * spin_term +
            0.25 * upright_term +
            0.10 * forward_term
        )

        # squash to [-1, 1]
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

    def arm_swing(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_arm_swing(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    def energy_efficient(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_energy_efficient(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)
    
    def crawler(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_crawler(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)
    
    def beyblade(obs, action, reward, term, trunc, info):
        base = _base_reward(obs, action, reward, term, trunc, info)
        style = _style_beyblade(obs, action, reward, term, trunc, info)
        return _apply_style(base, style)

    components = [
        RewardComponent(name="base", fn=_base_reward),
        RewardComponent(name="jumpy", fn=jumpy),
        RewardComponent(name="crouch", fn=crouch),
        RewardComponent(name="arm_swing", fn=arm_swing),
        RewardComponent(name="crawler", fn=crawler),
        RewardComponent(name="beyblade", fn=beyblade),
    ]

    return RewardConfig(env_id=env_id, components=components)