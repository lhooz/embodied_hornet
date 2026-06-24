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
import haiku as hk

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
    dist_sq = jnp.sum(disp**2, axis=-1) + 2.5e-3  # Stabilize: floor at 2.5e-3 (5cm physical)
    
    # Only consider obstacles/walls (membrane potential > 0)
    active_mask = jnp.maximum(SOG_v_mem, 0.0)  # (..., N, N) or (N, N)
    
    # Local influence cutoff (40cm down to 5cm) for long-range steering
    # We use a C2-continuous smooth envelope to give a very gentle warning at long range
    d_max = 0.40
    d_min = 0.05
    dist = jnp.sqrt(dist_sq)
    u = jnp.maximum((d_max - dist) / (d_max - d_min), 0.0)
    smooth_envelope = u ** 3
    repulsion = active_mask * (0.05 / dist_sq) * smooth_envelope
    
    # Sum up all repulsive force contributions
    force = jnp.sum(disp * repulsion[..., None], axis=(-3, -2))  # (..., 2)
    
    # Safety Valve: Clip repulsive force magnitude to 8.0 to bound forces and gradients
    force_magnitude = jnp.sqrt(jnp.sum(force**2, axis=-1) + 1e-8)
    scale = jnp.where(force_magnitude > 8.0, 8.0 / force_magnitude, 1.0)
    force = force * scale[..., None]
    
    return force

def policy_network_icnn(x, target_state=None, action_noise=None, SOG_v_mem=None, K_repel=0.0,
                        emd_signals=None, K_flow=0.0, K_loom=0.0, robot_pos_slam=None,
                        dynamic_gains=True, instar_belief=None, K_instar=1.0):
    """
    Full Policy Pipeline: Brain -> Muscles (Enhanced with SOG, EMD, and Instar memory avoidance).
    
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
 
    # 1b. Predict dynamic neuromodulatory gains (using x, the raw 8D physical state)
    # This must be done outside the emd_signals conditional so that Haiku can
    # initialize the neuromodulator parameters during ac.init (where emd_signals=None).
    if dynamic_gains:
        net_K = hk.Sequential([
            hk.Linear(16, name="neuromod_1"), jax.nn.tanh,
            hk.Linear(2, name="neuromod_2")
        ])
        raw_K = net_K(x)
        
        # Map to positive bounded ranges:
        scale_flow = K_flow
        scale_loom = K_loom
        K_flow_dyn = jax.nn.sigmoid(raw_K[..., 0]) * 2.0 * scale_flow
        K_loom_dyn = jax.nn.sigmoid(raw_K[..., 1]) * 2.0 * scale_loom

    # Reconstruct robot_pos_slam and sensor heading if None/from observations
    # Invert symlog from observation x: index 0,1 are relative position, index 2 is body pitch
    sign_x = jnp.sign(x[..., :2])
    abs_x = jnp.abs(x[..., :2])
    rel_pos = sign_x * jnp.expm1(abs_x)
    robot_pos_slam = rel_pos + 1.0
    
    sign_th = jnp.sign(x[..., 2])
    abs_th = jnp.abs(x[..., 2])
    pitch_angle = sign_th * jnp.expm1(abs_th)
    heading = pitch_angle - 1.0  # sensor heading

    # Saturate the goal-seeking brain forces first (dimensionless ratios in [-1, 1])
    u_forces_saturated = jnp.tanh(u_forces_newtons / ScaleConfig.CONTROL_SCALE)

    # 2. Inject Hassenstein-Reichardt EMD reflexes (Dorsal pathway — LPTC)
    #    emd_signals[..., 0] = HS centering: left-vs-right flow differential
    #    emd_signals[..., 1] = LGMD looming: total unsigned flow energy
    if emd_signals is not None:
        centering = emd_signals[..., 0]
        looming   = emd_signals[..., 1]
        
        # Proximity gating for EMD to restrict it to close-up range (< 15cm) in the visual FOV
        if SOG_v_mem is not None:
            N = SOG_v_mem.shape[-1]
            coords = jnp.linspace(0.0, 2.0, N)
            X, Z = jnp.meshgrid(coords, coords, indexing='ij')
            cell_positions = jnp.stack([X, Z], axis=-1)  # (N, N, 2)
            
            disp = cell_positions - robot_pos_slam[..., None, None, :]
            dist_sq = jnp.sum(disp**2, axis=-1) + 2.5e-3
            dist = jnp.sqrt(dist_sq)
            
            # Compute relative angle of cells to heading
            dx = disp[..., 0]
            dz = disp[..., 1]
            cell_phi = jnp.atan2(dz, dx)
            
            h_expanded = heading[..., None, None]
            delta_phi = jnp.mod(cell_phi - h_expanded + jnp.pi, 2 * jnp.pi) - jnp.pi
            
            # Soft FOV mask: event camera FOV is 90° (+/- 45° or +/- pi/4 rad)
            # Sigmoid transition is smooth and C2 differentiable for SHAC
            fov_mask = jax.nn.sigmoid(20.0 * (jnp.pi / 4 - jnp.abs(delta_phi)))
            
            active_mask = jnp.maximum(SOG_v_mem, 0.0)
            
            # EMD Proximity Envelope: active under 15cm
            d_max_emd = 0.15
            d_min_emd = 0.05
            u_emd = jnp.maximum((d_max_emd - dist) / (d_max_emd - d_min_emd), 0.0)
            cell_proximity = active_mask * (u_emd ** 2) * fov_mask
            emd_gate = jnp.max(cell_proximity, axis=(-2, -1))
        else:
            emd_gate = 1.0
            
        if dynamic_gains:
            # HS-cell centering: pitch torque correction (steer away from approaching side)
            u_forces_saturated = u_forces_saturated.at[..., 2].add(emd_gate * K_flow_dyn * centering)
            # LGMD looming escape: forward deceleration (brake when total expansion is high)
            u_forces_saturated = u_forces_saturated.at[..., 0].add(emd_gate * -K_loom_dyn * looming)
        else:
            # HS-cell centering: pitch torque correction (steer away from approaching side)
            u_forces_saturated = u_forces_saturated.at[..., 2].add(emd_gate * K_flow * centering)
            # LGMD looming escape: forward deceleration (brake when total expansion is high)
            u_forces_saturated = u_forces_saturated.at[..., 0].add(emd_gate * -K_loom * looming)

    # 3. Inject SOG repulsive forces (Ventral pathway)
    if SOG_v_mem is not None:
        f_repel = compute_sog_repulsive_force(robot_pos_slam, SOG_v_mem)
        # Add repulsive forces to Fx (index 0) and Fz (index 1) in ratio space
        u_forces_saturated = u_forces_saturated.at[..., 0].add(K_repel * f_repel[..., 0] / ScaleConfig.CONTROL_SCALE[0])
        u_forces_saturated = u_forces_saturated.at[..., 1].add(K_repel * f_repel[..., 1] / ScaleConfig.CONTROL_SCALE[1])

    # 4. Inject Instar visual-spatial memory forces (Ventral stream feedforward)
    if instar_belief is not None:
        u_forces_saturated = u_forces_saturated + K_instar * instar_belief

    # 5. Apply Action Noise (Post-Tanh) and Clip to biological limits [-1.0, 1.0]
    if action_noise is not None:
        u_forces_saturated = u_forces_saturated + action_noise
    
    u_forces_saturated = jnp.clip(u_forces_saturated, -1.0, 1.0)
    
    # 6. THE MUSCLES (Map Forces -> Kinematics)
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces_saturated)
    modulations_vector = jnp.stack(mod_tuple, axis=-1)

    net_forces = u_forces_saturated * ScaleConfig.CONTROL_SCALE
    u_brain_saturated = jnp.tanh(u_forces_newtons / ScaleConfig.CONTROL_SCALE)
    brain_goal_forces = u_brain_saturated * ScaleConfig.CONTROL_SCALE
    
    if instar_belief is not None:
        instar_forces = K_instar * instar_belief * ScaleConfig.CONTROL_SCALE
    else:
        instar_forces = jnp.zeros_like(net_forces)
        
    stacked_forces = jnp.stack([net_forces, brain_goal_forces, instar_forces], axis=-2)
    
    return modulations_vector, stacked_forces


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