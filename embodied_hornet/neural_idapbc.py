"""
embodied_hornet/neural_idapbc.py

Integration extensions to hornetRL's IDA-PBC controller.
Imports all base classes from the hornetRL submodule and adds:
  - Hassenstein-Reichardt EMD reflexes (Dorsal pathway — LPTC)
  - Spiking Occupancy Grid (SOG) Artificial Potential Field (Ventral pathway)
  - differentiable_attention_gate(): DNAG smooth blending based on SLAM surprise
"""
import jax
import jax.numpy as jnp
from functools import partial

# --- ALL BASE CLASSES FROM hornetRL SUBMODULE (unmodified) ---
from hornetRL.neural_idapbc import (
    ScaleConfig,
    ICNN,
    NeuralIDAPBC_ICNN,
    unpack_action,
)
from hornetRL.neural_cpg import BiologicalKinematicMap

@partial(jax.jit, static_argnames=['map_size'])
def compute_sog_repulsive_force(robot_pos_slam, SOG_v_mem, map_size=2.0):
    """
    Computes the 2D repulsive force vector from the Spiking Occupancy Grid (SOG) membrane potential.
    """
    N = SOG_v_mem.shape[-1]  # grid size (e.g., 50)
    coords = jnp.linspace(0.0, map_size, N)
    X, Z = jnp.meshgrid(coords, coords, indexing='ij')
    cell_positions = jnp.stack([X, Z], axis=-1)  # (N, N, 2)
    
    # Compute displacements from robot pos (..., N, N, 2)
    disp = robot_pos_slam[..., None, None, :] - cell_positions
    dist_sq = jnp.sum(disp**2, axis=-1) + 1e-2  # Stabilize: floor at 1e-2 (10cm physical)
    
    # Only consider obstacles/walls (membrane potential > 0)
    active_mask = jnp.maximum(SOG_v_mem, 0.0)  # (..., N, N) or (N, N)
    
    # Local influence cutoff (e.g. 0.25 meters) to prevent long-range gravitation-like pulling
    # We use a C1-continuous smooth envelope (1.0 - d/d_max)^2 to avoid gradient discontinuities in SHAC
    d_max = 0.25
    dist = jnp.sqrt(dist_sq)
    smooth_envelope = jnp.maximum(1.0 - dist / d_max, 0.0) ** 2
    repulsion = active_mask * (1.0 / dist_sq) * smooth_envelope
    
    # Sum up all repulsive force contributions
    force = jnp.sum(disp * repulsion[..., None], axis=(-3, -2))  # (..., 2)
    
    # Safety Valve: Clip repulsive force magnitude to 10.0 to bound forces and gradients
    force_magnitude = jnp.sqrt(jnp.sum(force**2, axis=-1) + 1e-8)
    scale = jnp.where(force_magnitude > 10.0, 10.0 / force_magnitude, 1.0)
    force = force * scale[..., None]
    
    return force

def policy_network_icnn(x, target_state=None, action_noise=None, SOG_v_mem=None, K_repel=0.0,
                        emd_signals=None, K_flow=0.0, K_loom=0.0, robot_pos_slam=None):
    """
    Full Policy Pipeline: Brain -> Muscles (Enhanced with SOG & EMD avoidance).
    
    Maps normalized observations to biological CPG modulation parameters.
    
    The Dorsal pathway now uses Hassenstein-Reichardt EMD signals from the
    1D ommatidial array (event camera), replacing the previous static ToF
    proximity differential.  Two LPTC reflexes are injected:
      - HS centering:  pitch torque to steer away from approaching side
      - LGMD looming:  forward deceleration when total expansion is high
    """
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    
    # Apply "Volume Knobs" (Sensitivity Gains)
    x_in = x * ScaleConfig.OBS_SCALE

    # 1. THE BRAIN (Compute Generalized Forces)
    brain = NeuralIDAPBC_ICNN(target_state)
    u_forces_newtons = brain(x_in)

    # 2. Inject Hassenstein-Reichardt EMD reflexes (Dorsal pathway — LPTC)
    #    emd_signals[..., 0] = HS centering: left-vs-right flow differential
    #    emd_signals[..., 1] = LGMD looming: total unsigned flow energy
    if emd_signals is not None:
        centering = emd_signals[..., 0]
        looming   = emd_signals[..., 1]
        # HS-cell centering: pitch torque correction (steer away from approaching side)
        u_forces_newtons = u_forces_newtons.at[..., 2].add(K_flow * centering)
        # LGMD looming escape: forward deceleration (brake when total expansion is high)
        u_forces_newtons = u_forces_newtons.at[..., 0].add(-K_loom * looming)

    # 3. Inject SOG repulsive forces (Ventral pathway)
    if SOG_v_mem is not None:
        if robot_pos_slam is None:
            # Fallback: Reconstruct relative position from symlog observation
            # x is symlog(obs_robot) where obs_robot[..., :2] = slam_pos_t - target_xy
            # Since target_xy is at the origin of the relative observation space,
            # we invert the symlog to retrieve target-relative position.
            sign_x = jnp.sign(x[..., :2])
            abs_x = jnp.abs(x[..., :2])
            rel_pos = sign_x * jnp.expm1(abs_x)
            # Assume target is at origin (0, 0), so slam_pos_t = rel_pos
            # robot_pos_slam = slam_pos_t + 1.0 offset
            robot_pos_slam = rel_pos + 1.0
            
        f_repel = compute_sog_repulsive_force(robot_pos_slam, SOG_v_mem)
        # Add repulsive forces to Fx (index 0) and Fz (index 1)
        u_forces_newtons = u_forces_newtons.at[..., 0].add(K_repel * f_repel[..., 0])
        u_forces_newtons = u_forces_newtons.at[..., 1].add(K_repel * f_repel[..., 1])

    # 4. Normalize using CONTROL_SCALE
    raw_ratio = u_forces_newtons / ScaleConfig.CONTROL_SCALE
    u_forces_saturated = jnp.tanh(raw_ratio) 

    # 5. Apply Action Noise (Post-Tanh)
    if action_noise is not None:
        u_forces_saturated = u_forces_saturated + action_noise
        u_forces_saturated = jnp.clip(u_forces_saturated, -1.0, 1.0)
    
    # 6. THE MUSCLES (Map Forces -> Kinematics)
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces_saturated)
    
    modulations_vector = jnp.stack(mod_tuple, axis=-1)
    
    return modulations_vector, u_forces_newtons


# ==============================================================================
# DIFFERENTIABLE NEUROMODULATORY ATTENTION GATE (DNAG)
# ==============================================================================
def differentiable_attention_gate(surprise, policy_mods, hover_mods, gamma=15.0):
    """
    Differentiable Neuromodulatory Attention Gate (DNAG).
    Smoothly blends normal policy modulations with passivity-based hover modulations
    based on the Surprise metric from neuro-symbolic-slam, keeping the entire
    pipeline fully differentiable.
    """
    alpha = jax.nn.sigmoid(gamma * (surprise - 0.30))
    alpha = jnp.expand_dims(alpha, axis=-1)   # scalar→(1,) or (B,)→(B,1)

    blended_mods = (1.0 - alpha) * policy_mods + alpha * hover_mods

    return blended_mods, alpha