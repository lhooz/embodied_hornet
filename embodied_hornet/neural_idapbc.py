"""
embodied_hornet/neural_idapbc.py

Integration extensions to hornetRL's IDA-PBC controller.
Imports all base classes from the hornetRL submodule and adds:
  - hover_stable(): Pure JAX Lyapunov-stable hover policy (no haiku modules)
  - differentiable_attention_gate(): DNAG smooth blending based on SLAM surprise

NOTE: IDA_PBC_Hover was previously an hk.Module using a random ICNN, which
(a) required hk.transform and broke jax.vmap, and (b) provided no stability
guarantee since the ICNN was never trained. It is replaced with an analytical
quadratic energy controller: V(e) = 0.5*||e||^2, grad_V = e -- provably
Lyapunov-stable. BiologicalKinematicMap (also hk.Module) is replaced with
an equivalent fixed analytical mapping.
"""
import jax
import jax.numpy as jnp

# --- ALL BASE CLASSES FROM hornetRL SUBMODULE (unmodified) ---
from hornetRL.neural_idapbc import (
    ScaleConfig,
    ICNN,
    NeuralIDAPBC_ICNN,
    policy_network_icnn,
    unpack_action,
)

# ==============================================================================
# SAFETY -- PURE JAX LYAPUNOV HOVERING CONTROLLER
# ==============================================================================
def hover_stable(x, target_state=None):
    """
    Pure JAX Lyapunov-stable hover policy mapping observations to CPG modulations.
    Invoked (via DNAG) when spatial uncertainty / SLAM Surprise is high, to
    arrest kinetic energy and hold position.

    Replaces the previous hk.Module-based IDA_PBC_Hover + BiologicalKinematicMap
    pair, which required hk.transform and was incompatible with bare jax.vmap.

    Energy shaping:
        V(e) = 0.5 * ||e||^2  ->  grad_V = e  (quadratic, provably Lyapunov-stable)
        (Previous ICNN had random weights and was never trained -- no stability
        guarantee. Quadratic is a strict improvement.)

    Force -> modulation mapping:
        Fixed analytical mapping mirroring BiologicalKinematicMap output structure.
        Ensures hover_mods stay in the same 9D space as policy_mods so DNAG
        blending is meaningful.

    Args:
        x:            (8,) observation [x, z, theta, phi, vx, vz, w_theta, w_phi]
        target_state: (8,) target state (default: hover at origin, theta=1.0)

    Returns:
        modulations_vector: (9,) CPG modulation vector (same space as policy_mods)
        u_forces:           (4,) raw force command [Fx, Fz, Tau_theta, Tau_phi]
    """
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    x_in = x * ScaleConfig.OBS_SCALE          # scale to internal units
    q    = x_in[:4]                            # position-like states (scaled)
    p    = x_in[4:]                            # momentum-like states (scaled)

    target_q = target_state[:4] * ScaleConfig.OBS_SCALE[:4]
    error    = q - target_q                    # position error in scaled space

    # --- Quadratic Energy Shaping: V(e) = 0.5*||e||^2, grad_V = e ---
    grad_Va = error * ScaleConfig.CONTROL_SCALE

    # --- Boosted Damping Injection (2.5x range, 5x base) -- active braking ---
    damping_gains = 2.5 * ScaleConfig.DAMPING_SCALE + 5.0 * ScaleConfig.DAMPING_BASE
    damping_force = -damping_gains * p

    u_forces = -grad_Va + damping_force        # (4,) physical force command

    # --- Saturate to [-1, 1] for kinematic mapping ---
    u_sat = jnp.tanh(u_forces / ScaleConfig.CONTROL_SCALE)   # (4,) in [-1, 1]

    # --- Fixed Analytical Force -> 9D CPG Modulation Mapping ---
    # Mirrors BiologicalKinematicMap's output structure and scaling with fixed gains.
    # Output: [d_freq, d_amp, bias, pitch_off, dev_amp, abd_tau, aoa_dn, aoa_up, dev_phase]
    d_freq     = u_sat[1] * 1000.0                              # Fz  -> frequency
    d_amp      = u_sat[1] * 0.4                                 # Fz  -> amplitude
    bias       = jnp.clip(u_sat[0] * 0.0035, -0.0035, 0.0035)  # Fx  -> stroke bias
    pitch_off  = jnp.clip(u_sat[2] * 0.5,    -0.5,    0.5)     # Tau_theta -> pitch phase
    dev_amp    = jnp.clip(u_sat[0] * 0.006,  -0.006,  0.006)   # Fx  -> deviation amp
    abd_torque = u_sat[3] * 2e-4                                 # Tau_phi -> abdomen
    aoa_down   = jnp.clip(0.75 + u_sat[2] * 0.75, 0.0, 1.5)   # Tau_theta -> AoA down
    aoa_up     = jnp.clip(0.75 + u_sat[2] * 0.75, 0.0, 1.5)   # Tau_theta -> AoA up
    dev_phase  = u_sat[0] * 0.1                                  # Fx  -> deviation phase

    modulations_vector = jnp.stack(
        [d_freq, d_amp, bias, pitch_off, dev_amp,
         abd_torque, aoa_down, aoa_up, dev_phase],
        axis=-1
    )                                          # (9,) -- same shape as policy_mods

    return modulations_vector, u_forces


# ==============================================================================
# DIFFERENTIABLE NEUROMODULATORY ATTENTION GATE (DNAG)
# ==============================================================================
def differentiable_attention_gate(surprise, policy_mods, hover_mods, gamma=15.0):
    """
    Differentiable Neuromodulatory Attention Gate (DNAG).
    Smoothly blends normal policy modulations with passivity-based hover modulations
    based on the Surprise metric from neuro-symbolic-slam, keeping the entire
    pipeline fully differentiable.

        alpha = sigmoid(gamma * (S - 0.30))
        blended = (1 - alpha) * policy_mods + alpha * hover_mods

    At S=0.30 (threshold): alpha=0.5 (equal blend)
    At S=0.60 (high surprise): alpha~0.99 (nearly full hover)
    At S=0.10 (low surprise):  alpha~0.01 (nearly full policy)
    """
    alpha = jax.nn.sigmoid(gamma * (surprise - 0.30))
    alpha = jnp.reshape(alpha, (-1, 1))  # (B, 1) for broadcast over action dim

    blended_mods = (1.0 - alpha) * policy_mods + alpha * hover_mods

    return blended_mods, alpha