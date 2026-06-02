import jax
import jax.numpy as jnp
import haiku as hk

# Import CPG kinematic and muscle mapping modules from hornetRL sibling repo
from hornetRL.neural_cpg import BiologicalKinematicMap

# ==============================================================================
# 0. SCALING CONFIGURATION (INTERNAL PHYSICS TUNING)
# ==============================================================================
class ScaleConfig:
    """
    Internal scaling parameters for the controller physics.
    Defines the magnitude of control authority and damping relative to physical units.
    """
    # Control Authority: [Fx (N), Fz (N), Tau_theta (Nm), Tau_phi (Nm)]
    CONTROL_SCALE = jnp.array([0.05, 0.05, 5e-4, 2.0e-4])

    # Damping Baseline: Low drag for efficient flight
    DAMPING_BASE = jnp.array([0.002, 0.002, 1.0e-5, 1.0e-5])

    # Damping Range: Allows strong braking (linear) and precise attitude control (angular)
    DAMPING_SCALE = jnp.array([0.05, 0.05, 5e-4, 2.0e-4])

    # Define Scale (Sensitivity Multipliers)
    #                    [  x,    z,   th,  phi,  vx,  vz,w_th, w_phi]
    OBS_SCALE = jnp.array([50.0, 50.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.1])

# ==============================================================================
# 1. INPUT CONVEX NEURAL NETWORK (ICNN)
# ==============================================================================
class ICNN(hk.Module):
    """
    Input Convex Neural Network (ICNN) implementation.
    
    Used as a learnable potential energy function V(x). The network architecture 
    guarantees convexity with respect to the input by enforcing non-negative 
    weights on the pass-through connections and using convex activation functions.
    """
    def __init__(self, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def __call__(self, x):
        # Initial transformation (can be negative)
        u = hk.Linear(self.hidden_dim, name="input_layer")(x)
        z = jax.nn.softplus(u) ** 2 
        
        # Convex layers (enforce non-negative weights on z)
        input_injection = hk.Linear(self.hidden_dim, name="hidden_inject")(x)
        
        w_z_init = hk.initializers.VarianceScaling()
        w_z = hk.get_parameter("w_z_1", shape=[self.hidden_dim, self.hidden_dim], init=w_z_init)
        
        # z_{k+1} = sigma(z_k * W_z + x * W_x + b)
        # Note: We take absolute value of W_z to ensure convexity
        z = jax.nn.softplus(jnp.dot(z, jnp.abs(w_z)) + input_injection) ** 2
        
        w_out = hk.get_parameter("w_out", shape=[self.hidden_dim, self.output_dim], init=w_z_init)
        energy = jnp.dot(z, jnp.abs(w_out))
        
        # Add quadratic regularization to ensure strict convexity
        energy = energy + 0.005 * jnp.sum(x**2)
        
        return jnp.squeeze(energy)

# ==============================================================================
# 2. BRAIN: IDA-PBC CONTROLLER
# ==============================================================================
class NeuralIDAPBC_ICNN(hk.Module):
    """
    Neural Interconnection and Damping Assignment - Passivity Based Controller.
    
    This module shapes the closed-loop energy of the system using an ICNN 
    to drive the state error to zero. It computes control forces as:
    u = -grad(V_shaped(error)) - R(x) * velocity
    """
    def __init__(self, target_state):
        super().__init__()
        raw_target_q = target_state[:4]
        self.target_q = raw_target_q * ScaleConfig.OBS_SCALE[:4]
        self.icnn = ICNN()
        
    def __call__(self, x):
        # Split state into Generalized Coordinates (q) and Momentum/Velocity (p)
        # Note: x is already scaled by the environment wrapper
        q = x[..., :4] 
        p = x[..., 4:]
        
        # 1. Energy Shaping (Potential Field)
        # Compute error in the normalized latent space
        error = q - self.target_q
        
        def energy_fn(e): return jnp.sum(self.icnn(e))
        
        # Calculate the gradient of the shaped energy function
        raw_grad = jax.grad(energy_fn)(error)
        
        # Scale the gradient to physical force units (Newtons/Nm)
        grad_Va = raw_grad * ScaleConfig.CONTROL_SCALE
        
        # 2. Damping Injection (Dissipation)
        # Predict dynamic damping gains based on system state
        net_R = hk.Sequential([hk.Linear(32), jax.nn.tanh, hk.Linear(4)])
        raw_damp = net_R(x)
        
        damping_gains = (jax.nn.softplus(raw_damp) * ScaleConfig.DAMPING_SCALE) + ScaleConfig.DAMPING_BASE
        damping_force = -damping_gains * p
        
        # Total Control Law: u = -dV/dq - R * p
        return -grad_Va + damping_force

# ==============================================================================
# 2B. SAFETY: IDA-PBC LYAPUNOV HOVERING CONTROLLER
# ==============================================================================
class IDA_PBC_Hover(hk.Module):
    """
    Dedicated passivity-based hovering controller with boosted damping injection
    to act as a Lyapunov-stable active brake during spatial re-localization.
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
        
        # Boosted Damping Injection (Active Braking to halt and hold altitude)
        # We multiply the damping range and base to enforce passivity rapidly
        damping_gains = (2.5 * ScaleConfig.DAMPING_SCALE) + 5.0 * ScaleConfig.DAMPING_BASE
        damping_force = -damping_gains * p
        
        return -grad_Va + damping_force

# ==============================================================================
# 3. FULL POLICY WRAPPER
# ==============================================================================
def policy_network_icnn(x, target_state=None, action_noise=None):
    """
    Full Policy Pipeline: Brain -> Muscles.
    
    Maps observations to biological CPG modulation parameters.
    """
    # Default hover target if none provided
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    
    # Apply Obs scale
    x_in = x * ScaleConfig.OBS_SCALE

    # 1. THE BRAIN (Compute Generalized Forces)
    brain = NeuralIDAPBC_ICNN(target_state)
    u_forces_newtons = brain(x_in)

    # 2. Normalize using CONTROL_SCALE
    raw_ratio = u_forces_newtons / ScaleConfig.CONTROL_SCALE
    
    # 3. Apply Soft Saturation
    u_forces_saturated = jnp.tanh(raw_ratio) 

    # 4. Inject Action Noise
    if action_noise is not None:
        u_forces_saturated = u_forces_saturated + action_noise
        u_forces_saturated = jnp.clip(u_forces_saturated, -1.0, 1.0)
    
    # 5. THE MUSCLES (Map Forces -> Kinematics)
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces_saturated)
    
    # Stack outputs: [d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_dn, aoa_up, dev_phase]
    modulations_vector = jnp.stack(mod_tuple, axis=-1)
    
    return modulations_vector, u_forces_newtons

# ==============================================================================
# 3B. LYAPUNOV STABLE HOVERING INTERFACE
# ==============================================================================
def hover_stable(x, target_state=None):
    """
    Lyapunov-stable active hovering policy mapping observations to CPG modulations.
    Invoked when spatial uncertainty/Surprise is high to arrest kinetic energy.
    """
    if target_state is None:
        target_state = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    
    # Observe scaled inputs
    x_in = x * ScaleConfig.OBS_SCALE
    
    # Brain: Dedicated ICNN Hover formulation
    brain = IDA_PBC_Hover(target_state)
    u_forces = brain(x_in)
    
    # Saturation
    u_forces_saturated = jnp.tanh(u_forces / ScaleConfig.CONTROL_SCALE)
    
    # Muscle Mapping
    muscles = BiologicalKinematicMap()
    mod_tuple = muscles(u_forces_saturated)
    modulations_vector = jnp.stack(mod_tuple, axis=-1)
    
    return modulations_vector, u_forces

# ==============================================================================
# 3C. DIFFERENTIABLE ATTENTION GATING (DNAG)
# ==============================================================================
def differentiable_attention_gate(surprise, policy_mods, hover_mods, gamma=15.0):
    """
    Differentiable Neuromodulatory Attention Gate (DNAG).
    Smoothly blends normal policy modulations with passivity-based hover modulations
    based on the Surprise metric, keeping the entire pipeline fully differentiable.
    
    $$\alpha = \sigma(\gamma \cdot (S - 0.30))$$
    """
    # Compute continuous gating factor alpha
    alpha = jax.nn.sigmoid(gamma * (surprise - 0.30))
    alpha = jnp.reshape(alpha, (-1, 1))  # Shape (B, 1) for safe broadcast
    
    # Blend kinematics continuously
    blended_mods = (1.0 - alpha) * policy_mods + alpha * hover_mods
    
    return blended_mods, alpha

# ==============================================================================
# 4. HELPER: UNPACK ACTION
# ==============================================================================
def unpack_action(action_vector):
    """
    Utility to decompose the stacked action vector into named components.
    """
    d_freq = action_vector[..., 0]
    d_amp  = action_vector[..., 1]
    d_bias = action_vector[..., 2]
    pitch_off = action_vector[..., 3]
    dev_amp = action_vector[..., 4]
    abd_tau = action_vector[..., 5]
    aoa_down = action_vector[..., 6]
    aoa_up = action_vector[..., 7]
    dev_phase = action_vector[..., 8] 
    
    return (d_freq, d_amp, d_bias, pitch_off, dev_amp, abd_tau, aoa_down, aoa_up, dev_phase)