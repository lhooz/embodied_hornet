"""
embodied_hornet/neural_idapbc.py

Integration extensions to hornetRL's IDA-PBC controller.
Imports all base classes from the hornetRL submodule and adds:
  - IDA_PBC_Hover: boosted-damping passivity controller for spatial re-localisation
  - hover_stable(): Lyapunov-stable hover policy interface
  - differentiable_attention_gate(): DNAG smooth blending based on SLAM surprise
"""
import jax
import jax.numpy as jnp
import haiku as hk

# --- ALL BASE CLASSES FROM hornetRL SUBMODULE (unmodified) ---
from hornetRL.neural_idapbc import (
    ScaleConfig,
    ICNN,
    NeuralIDAPBC_ICNN,
    policy_network_icnn,
    unpack_action,
)
from hornetRL.neural_cpg import BiologicalKinematicMap

# ==============================================================================
# NEW: SAFETY — IDA-PBC LYAPUNOV HOVERING CONTROLLER
# ==============================================================================
class IDA_PBC_Hover(hk.Module):
    """
    Dedicated passivity-based hovering controller with boosted damping injection
    to act as a Lyapunov-stable active brake during spatial re-localization.

    Unlike NeuralIDAPBC_ICNN (which learns dynamic damping), this controller
    uses fixed, amplified damping to guarantee kinetic energy dissipation
    regardless of the learned policy state.
    """
    def __init__(self, target_state):
        super().__init__()
        raw_target_q = target_state[:4]
        self.target_q = raw_target_q * ScaleConfig.OBS_SCALE[:4]
        self.icnn = ICNN(name="hover_icnn")

    def __call__(self, x):
        q = x[..., :4]
        p = x[..., 4:]
        error = q - self.target_q

        # Energy shaping gradient
        def energy_fn(e): return jnp.sum(self.icnn(e))
        raw_grad = jax.grad(energy_fn)(error)
        grad_Va = raw_grad * ScaleConfig.CONTROL_SCALE

        # Boosted Damping Injection (2.5x range, 5x base) — active braking
        damping_gains = (2.5 * ScaleConfig.DAMPING_SCALE) + 5.0 * ScaleConfig.DAMPING_BASE
        damping_force = -damping_gains * p

        return -grad_Va + damping_force


# ==============================================================================
# NEW: LYAPUNOV STABLE HOVERING INTERFACE
# ==============================================================================
def hover_stable(x, target_state=None):
    """
    Lyapunov-stable active hovering policy mapping observations to CPG modulations.
    Invoked when spatial uncertainty/Surprise is high to arrest kinetic energy.
    """
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    x_in = x * ScaleConfig.OBS_SCALE

    brain = IDA_PBC_Hover(target_state)
    u_forces = brain(x_in)

    u_forces_saturated = jnp.tanh(u_forces / ScaleConfig.CONTROL_SCALE)

    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces_saturated)
    modulations_vector = jnp.stack(mod_tuple, axis=-1)

    return modulations_vector, u_forces


# ==============================================================================
# NEW: DIFFERENTIABLE NEUROMODULATORY ATTENTION GATE (DNAG)
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