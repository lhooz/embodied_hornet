import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Any

# --- MODULES FROM hornetRL SIBLING REPO ---
from hornetRL.fly_system import FlappingFlySystem, PhysParams
from hornetRL.neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from hornetRL.neural_idapbc import unpack_action

# --- SENSOR FUNCTIONS FROM neuro-symbolic-slam SUBMODULE ---
# (resolved via sys.path set in embodied_hornet/__init__.py)
from sparse_forest import (
    generate_obstacles,
    obstacles_to_segments,
    compute_pixel_readings,
    compute_tof_distance,
    _precompute_barcode_tensors,
    _generate_surface_textures,
    THRESHOLD,
    N_PIXELS,
)

# ==============================================================================
# SLAM COORDINATE MAPPING
# ==============================================================================
# The hornet flies in a physical arena of ±ARENA_W metres.
# sparse_forest generates a 10m × 10m room with biologically-scaled obstacles
# (0.1-0.7m in 10m-space ≈ 1-7mm at hornet scale — twigs/stems).
# We linearly map hornet coordinates into this 10m room so the SLAM system
# receives sensor data in its native coordinate frame.
#
#   slam_coord = hornet_coord * SLAM_SCALE + SLAM_OFFSET
#   SLAM_SCALE = 5.0 / ARENA_W   →  [-ARENA_W, ARENA_W] maps to [0.5, 9.5] m
#   SLAM_OFFSET = 5.0            →  centres the hornet in the room
#
# The SLAM map, poses, and loop closures are all in 10m space.
# Only the Instar routing uses pose_belief converted BACK to hornet metres.
_SLAM_OFFSET = 5.0   # centre of the 10m room

# ==============================================================================
# ROBUST ENVIRONMENT
# ==============================================================================
class FlyEnv:
    """
    JAX-based environment handling the coupled Rigid Body and Fluid Surrogate dynamics.
    Implements SHAC-compatible step functions with automatic differentiation support.
    
    UPDATED: Multi-rate wrapper ingesting 3-DOF pose and visual streams at 460 Hz,
    implementing the Asymmetric Instar update rule ("Fast Learn, Slow Forget").
    """
    def __init__(self, config):
        """
        Args:
            config: A configuration class or object containing constants like
                    BASE_FREQ, TARGET_STATE, DT, ARENA_W, etc.
        """
        self.cfg = config

        self.phys = FlappingFlySystem(
            model_path='fluid.pkl',
            target_freq=config.BASE_FREQ
        )
        self.target = config.TARGET_STATE

        # --- Arena geometry for SLAM sensor generation ---
        # Scale factor: maps hornet ±ARENA_W metres into [0.5, 9.5] of the 10m room.
        # With ARENA_W=0.5 (1m×1m arena): slam_scale = 10.0×
        self._slam_scale  = _SLAM_OFFSET / getattr(config, 'ARENA_W', 0.5)
        self._obstacles   = None  # set by regenerate_arena()
        self._segments    = None
        self._tex_tensor  = None
        self.regenerate_arena(seed=42, quiet=True)  # initial room

    # ------------------------------------------------------------------
    def regenerate_arena(self, key=None, seed: int = None, quiet: bool = False):
        """
        Generates a new 10m × 10m obstacle room with fresh random obstacles
        and barcode wall textures.

        Called once at __init__ (fixed seed=42) and then at every forced
        episode reset in train.py so each episode has a new environment
        (the SLAM system is also reset externally when this is called).

        Args:
            key:   Optional JAX PRNGKey — used to derive seed if provided.
            seed:  Optional int seed — used directly if key is None.
            quiet: Suppress the print line (used on first call from __init__).
        """
        if key is not None:
            # Derive a deterministic int seed from a JAX key
            seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        elif seed is None:
            seed = 42

        rng = np.random.RandomState(seed)
        k_obs = jax.random.PRNGKey(rng.randint(0, 2**31))

        obstacles_jax    = generate_obstacles(k_obs)
        segments_jax     = obstacles_to_segments(obstacles_jax)
        obstacles_np     = np.array(obstacles_jax)
        room_seed        = int(rng.randint(0, 2**31))
        surface_textures = _generate_surface_textures(obstacles_np, room_seed)
        tex_tensor       = _precompute_barcode_tensors(surface_textures, obstacles_np)

        self._obstacles  = obstacles_jax
        self._segments   = segments_jax
        self._tex_tensor = tex_tensor
        if not quiet:
            print(f"---> Arena regenerated: 10m×10m, {len(obstacles_np)} obstacles "
                  f"(scale {self._slam_scale:.1f}×, seed {seed})")

    def reset(self, key, batch_size):
        """
        Resets the environment state, including robot state, fluid state, CPG, and
        the new Asymmetric Instar weights for visual-spatial routing.
        """
        k1, k2, k3, k4, k_shuffle = jax.random.split(key, 5)
        
        # =========================================================
        # 1. INIT ROBOT STATE (Curriculum Strategy)
        # =========================================================
        ratio = getattr(self.cfg, 'CURRICULUM_RATIO', 0.8)
        
        n_nominal = int(batch_size * ratio)
        n_chaos = batch_size - n_nominal
        
        # --- A. Nominal Group (Stable Hover Conditions, 80% of batch) ---
        # Small perturbation around hover theta=1.0: ±0.5 rad (28°).
        # This ensures the majority of agents start in a recoverable regime
        # where the IDA-PBC generates reasonable corrective torques.
        k1_n, k2_n = jax.random.split(k1)
        q_pos_nom = jax.random.uniform(k1_n, (n_nominal, 2), minval=-0.15, maxval=0.15)

        theta_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.5, maxval=0.5)
        theta_nom = theta_nom + 1.0  # → range [0.5, 1.5] rad, max error 0.5 rad from target

        phi_nom = jax.random.uniform(k2_n, (n_nominal, 1), minval=-0.1, maxval=0.1)
        phi_nom = phi_nom + 0.2

        q_ang_nom = jnp.concatenate([theta_nom, phi_nom], axis=-1)

        # --- B. Chaos Group (Recovery Training, 20% of batch) ---
        # Moderate tilt: ±π/2 (90°). Max theta error from 1.0 = 1.57 rad.
        # Original ±π allowed up to 3.14 rad error → AVel = 1000+ rad/s at
        # step 0 because the IDA-PBC saturates and applies maximum torque
        # continuously, causing rapid spin rather than controlled correction.
        k1_c, k2_c = jax.random.split(k2)
        k_theta, k_phi = jax.random.split(k2_c)
        q_pos_chaos = jax.random.uniform(k1_c, (n_chaos, 2), minval=-0.25, maxval=0.25)

        theta_chaos = jax.random.uniform(k_theta, (n_chaos, 1), minval=-jnp.pi/2, maxval=jnp.pi/2)
        theta_chaos = theta_chaos + 1.0  # → range [-0.57, 2.57] rad, max error 1.57 rad

        phi_chaos = jax.random.uniform(k_phi, (n_chaos, 1), minval=-0.5, maxval=0.5)
        phi_chaos = phi_chaos + 0.2

        q_ang_chaos = jnp.concatenate([theta_chaos, phi_chaos], axis=-1)

        
        # --- C. Combine & Velocity ---
        q_pos_ordered = jnp.concatenate([q_pos_nom, q_pos_chaos], axis=0)
        q_ang_ordered = jnp.concatenate([q_ang_nom, q_ang_chaos], axis=0)
        v_ordered     = jnp.zeros((batch_size, 4))
        
        perm = jax.random.permutation(k_shuffle, batch_size)
        
        q_pos = q_pos_ordered[perm]
        q_ang = q_ang_ordered[perm]
        v     = v_ordered[perm]
        
        robot_state_v = jnp.concatenate([q_pos, q_ang, v], axis=1)

        # =========================================================
        # 2. INIT OSCILLATOR (Random Phase)
        # =========================================================
        osc_state_single = OscillatorState.init(base_freq=self.cfg.BASE_FREQ) 
        def stack_batch(x): return jnp.stack([x] * batch_size)
        osc_state = jax.tree.map(stack_batch, osc_state_single)
        
        rand_phase = jax.random.uniform(k3, (batch_size,), minval=0.0, maxval=2*jnp.pi)
        osc_state = osc_state._replace(phase=rand_phase)

        # =========================================================
        # 3. CALCULATE WING POSE (Global -> Centered Local)
        # =========================================================
        zero_action = jnp.zeros((batch_size, 9)) 
        ret = jax.vmap(get_wing_kinematics)(osc_state, unpack_action(zero_action))
        k_angles = ret[0]
        k_rates  = ret[1]
        
        robot_state_p_dummy = jnp.concatenate([robot_state_v[:, :4], jnp.zeros((batch_size, 4))], axis=1)

        # =========================================================
        # 4. DOMAIN RANDOMIZATION (Physics Parameters)
        # =========================================================
        k_mass, k_com, k_hinge, k_st, k_joint = jax.random.split(k3, 5)
        
        mass_scale_th = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)
        mass_scale_ab = jax.random.uniform(k_mass, (batch_size,), minval=0.80, maxval=1.20)

        off_x_th = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)
        off_x_ab = jax.random.uniform(k_com, (batch_size,), minval=-0.002, maxval=0.002)

        h_x_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)
        h_z_noise = jax.random.uniform(k_hinge, (batch_size,), minval=-0.001, maxval=0.001)

        st_ang_noise = jax.random.uniform(k_st, (batch_size,), minval=-0.08, maxval=0.08)

        k_joint_keys = jax.random.split(k_joint, 3)
        k_hinge_scale = jax.random.uniform(k_joint_keys[0], (batch_size,), minval=0.7, maxval=1.3)
        b_hinge_scale = jax.random.uniform(k_joint_keys[1], (batch_size,), minval=0.7, maxval=1.3)
        phi_eq_off = jax.random.uniform(k_joint_keys[2], (batch_size,), minval=-0.1, maxval=0.1)

        phys_params = PhysParams(
            thorax_mass_scale=mass_scale_th,
            abd_mass_scale=mass_scale_ab,
            thorax_offset_x=off_x_th,
            abd_offset_x=off_x_ab,
            hinge_x_noise=h_x_noise,
            hinge_z_noise=h_z_noise,
            stroke_ang_noise=st_ang_noise,
            k_hinge_scale=k_hinge_scale,
            b_hinge_scale=b_hinge_scale,
            phi_equil_offset=phi_eq_off
        )

        active_props = jax.vmap(self.phys.robot.compute_props)(phys_params)
        wing_pose_global, _ = jax.vmap(self.phys.robot.get_kinematics)(robot_state_p_dummy, k_angles, k_rates, active_props)
        
        def get_centered_pose(r_state, w_pose_glob, bias_val, props):
            q = r_state[:4]
            theta = q[2]
            
            h_x = props.hinge_offset_x
            h_z = props.hinge_offset_z
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            hinge_glob_x = h_x * c_th - h_z * s_th
            hinge_glob_z = h_x * s_th + h_z * c_th
            
            total_st_ang = theta + props.stroke_plane_angle
            c_st, s_st = jnp.cos(total_st_ang), jnp.sin(total_st_ang)
            bias_glob_x = bias_val * c_st
            bias_glob_z = bias_val * s_st
            
            off_x = hinge_glob_x + bias_glob_x
            off_z = hinge_glob_z + bias_glob_z
            
            p_x = w_pose_glob[0] - (q[0] + off_x)
            p_y = w_pose_glob[1] - (q[1] + off_z)
            
            return jnp.array([p_x, p_y, w_pose_glob[2]])

        wing_pose_centered = jax.vmap(get_centered_pose)(robot_state_v, wing_pose_global, osc_state.bias, active_props)

        # =========================================================
        # 5. INIT FLUID STATE
        # =========================================================
        def init_fluid_fn(wp):
            return self.phys.fluid.init_state(wp[0], wp[1], wp[2])
            
        fluid_state = jax.vmap(init_fluid_fn)(wing_pose_centered)

        # =========================================================
        # 6. INIT INSTAR SYNAPTIC WEIGHTS (515 Perceptual -> 4 CPG policy target neurons)
        # =========================================================
        # We initialize with small random values to break symmetry.
        # 3 (Pose coordinate belief) + 256 (norm_csnn) + 256 (norm_stdp) = 515 input lines.
        k_instar = jax.random.split(k4)[0]
        W_instar = jax.random.uniform(k_instar, (batch_size, 515, 4), minval=0.01, maxval=0.05)

        return (robot_state_v, fluid_state, osc_state, active_props, W_instar)

    def step_batch(self, full_state, action_mods, step_idx=100):
        """
        Advances the simulation by one control step (Config.SIM_SUBSTEPS physics steps).
        Includes warmup ramping and velocity clamping for numerical stability.
        """
        robot_st, fluid_st, osc_st, active_props, W_instar = full_state
        
        # Define single agent step function for vmap/scan
        def single_agent_step(r, f, o, props, a):
            
            # --- Sub-stepping Loop (Physics Integration) ---
            def sub_step_fn(carry, _):
                curr_r, curr_f, curr_o = carry
                
                # 1. Oscillator Update (Steps by DT)
                o_next = step_oscillator(curr_o, unpack_action(a), self.cfg.DT)
                k_angles, k_rates, tau_abd, bias = get_wing_kinematics(o_next, unpack_action(a))
                
                action_data = (k_angles, k_rates, tau_abd, bias)

                # 2. Physics Update (Rigid Body + Fluid)
                (r_next_v, f_next), f_wing, f_nodal, wing_pose, hinge_marker = self.phys.step(
                    self.phys.fluid.params, (curr_r, curr_f), action_data, props, 0.0, self.cfg.DT
                )
                
                # --- Warmup Ramp & Stability ---
                ramp = jnp.clip(step_idx / self.cfg.WARMUP_STEPS, 0.0, 1.0)
                
                # 1. Velocity Reset: Pin fly during warmup
                v_reset = jnp.zeros(4) 
                r_next_v = jnp.where(step_idx < self.cfg.WARMUP_STEPS, r_next_v.at[4:].set(v_reset), r_next_v)
                
                # 2. Force Ramp: Scale nodal forces
                f_nodal_ramped = f_nodal * ramp
                
                # 3. Velocity Saturation: Safety clamp to prevent physics explosion
                v_limits = jnp.array([20.0, 20.0, 200.0, 200.0])
                v_current = r_next_v[4:]
                v_clamped = jnp.clip(v_current, -v_limits, v_limits)
                
                r_next_v = r_next_v.at[4:].set(v_clamped)
                
                # 4. Aux Loss Calculation Data
                tau_actual = f_wing[2] * ramp
                f_actual = jnp.array([f_wing[0]*ramp, f_wing[1]*ramp, tau_actual, 0.0])
                
                return (r_next_v, f_next, o_next), (f_actual, f_nodal_ramped, wing_pose, hinge_marker)

            # --- Execute Scan ---
            init_carry = (r, f, o)
            (final_r, final_f, final_o), (stacked_forces, stacked_nodals, stacked_poses, stacked_hinges) = jax.lax.scan(
                sub_step_fn, init_carry, None, length=self.cfg.SIM_SUBSTEPS
            )
            
            # --- Post-Processing ---
            avg_f_actual = jnp.mean(stacked_forces, axis=0)
            last_f_nodal = stacked_nodals[-1]
            last_wing_pose = stacked_poses[-1]
            last_hinge_marker = stacked_hinges[-1] 
            
            return final_r, final_f, final_o, avg_f_actual, last_f_nodal, last_wing_pose, last_hinge_marker

        # Checkpointing Optimization
        single_agent_step_remat = jax.checkpoint(single_agent_step)

        # Vectorize over batch
        r_n, f_n, o_n, f_act, f_nodal_b, w_pose_b, h_marker_b = jax.vmap(single_agent_step_remat)(
            robot_st, fluid_st, osc_st, active_props, action_mods
        )
        
        return (r_n, f_n, o_n, active_props, W_instar), f_act, f_nodal_b, w_pose_b, h_marker_b

    def ingest_perceptual_streams(self, full_state, pose_belief, visual_features, target_cpg_activity, eta=0.1, lam=0.001):
        """
        Routes and processes visual-spatial beliefs through the Asymmetric Instar update rule.
        Runs at the 460 Hz neural control rate to dynamically map high-dimensional visual 
        cues and topological spatial coordinates onto the policy inputs.
        
        Args:
            full_state: The current environment state tuple.
            pose_belief: 3-DOF spatial coordinates (x_hat, y_hat, th_hat) of shape (B, 3).
            visual_features: Tuple of (norm_csnn, norm_stdp) visual stream features, each shape (B, 256).
            target_cpg_activity: The active policy control outputs of shape (B, 4).
            eta: Fast learning rate for Instar update.
            lam: Slow forgetting decay rate.
            
        Returns:
            updated_state: The environment state with updated Instar weights.
            weighted_perceptual_belief: The instar-mapped 4D perceptual representation of shape (B, 4).
        """
        robot_st, fluid_st, osc_st, active_props, W_instar = full_state
        
        # 1. Synthesize the 515-dim Perceptual Input
        norm_csnn, norm_stdp = visual_features
        x_perceptual = jnp.concatenate([pose_belief, norm_csnn, norm_stdp], axis=-1)  # (B, 515)
        
        # 2. Batched Asymmetric Instar Update: dW = eta * y * (x - W) - lam * (1 - y) * W
        # W_instar shape: (B, 515, 4)
        # x_perceptual shape: (B, 515) -> expand to (B, 515, 1)
        # target_cpg_activity shape: (B, 4) -> expand to (B, 1, 4)
        x_exp = x_perceptual[..., None]
        y_exp = target_cpg_activity[..., None, :]
        
        delta_learn = (x_exp * y_exp) - W_instar * y_exp
        delta_forget = W_instar * (1.0 - y_exp)
        
        dW = eta * delta_learn - lam * delta_forget
        W_instar_next = W_instar + dW
        
        # 3. Project the perceptual inputs onto the policy using the updated weights
        # Out = W^T * x
        weighted_perceptual_belief = jnp.einsum('bip,bi->bp', W_instar_next, x_perceptual)  # (B, 4)
        
        updated_state = (robot_st, fluid_st, osc_st, active_props, W_instar_next)
        
        return updated_state, weighted_perceptual_belief

    def get_reward_metrics(self, robot_state, u_forces, reward_weights):
        """
        Calculates the scalar reward and detailed cost breakdown.
        Uses 'Honeypot' Precision Reward + 'Soft Barrier' Safety Penalty.
        """
        err = robot_state - self.target
        err_theta = jnp.mod(err[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi

        # --- 1. Position Metrics ---
        dist_sq = jnp.sum(err[:, :2]**2, axis=1)
        dist = jnp.sqrt(dist_sq + 1e-6)

        # --- 2. Precision Reward (The "Magnet") ---
        w_pos = reward_weights[:, 0]
        precision_kernel = 1.0 / (1.0 + 100.0 * dist_sq)
        rew_precision = w_pos * precision_kernel

        # --- 3. The "Nagger" (L1 Norm nudge) ---
        w_nudge = 1.0 * w_pos 
        linear_error = jnp.abs(err[:, 0]) + jnp.abs(err[:, 1])
        rew_linear_nudge = -w_nudge * linear_error

        # --- 4. Safety Penalty (The "Electric Fence") ---
        wall_limit = 0.20
        violation = jax.nn.relu(dist - wall_limit)
        penalty_scale = 20.0 * w_pos
        rew_safety = -penalty_scale * violation

        # --- 5. Other Dynamic Costs (Negative) ---
        w_th  = reward_weights[:, 1]
        w_ab  = reward_weights[:, 2]
        w_lv  = reward_weights[:, 3]
        w_av  = reward_weights[:, 4]
        w_eff = reward_weights[:, 5]

        loss_ang_thorax = err_theta**2
        loss_ang_abdomen = err[:, 3]**2
        loss_lin_vel = jnp.sum(err[:, 4:6]**2, axis=1)
        loss_ang_vel = jnp.sum(err[:, 6:8]**2, axis=1)
        loss_eff = jnp.sum(u_forces**2, axis=1)

        cost_others = (
            w_th  * loss_ang_thorax + 
            w_ab  * loss_ang_abdomen + 
            w_lv  * loss_lin_vel + 
            w_av  * loss_ang_vel + 
            w_eff * loss_eff
        )

        raw_reward = rew_precision + rew_linear_nudge + rew_safety - cost_others        
        scaled_reward = raw_reward * 0.02 
        
        metrics = {
            'rew': raw_reward,
            'pos': dist,
            'ang_th': loss_ang_thorax, 
            'ang_ab': loss_ang_abdomen,
            'vel_lin': loss_lin_vel,  
            'vel_ang': loss_ang_vel,  
            'ferr': loss_eff,
            'ang': loss_ang_thorax + loss_ang_abdomen
        }
        return scaled_reward, metrics

    # ------------------------------------------------------------------
    def compute_slam_sensors(
        self,
        robot_state_single: np.ndarray,
        prev_intensities: np.ndarray,
    ):
        """
        Computes event-camera, ToF, and kinematic odometry for ONE agent.

        Called OUTSIDE JAX JIT in the main training loop — not inside scan.
        The results are fed directly to SNNSLAMSystem.forward_step().

        Args:
            robot_state_single: (8,) numpy array  [x, z, theta, phi, vx, vz, w_theta, w_phi]
            prev_intensities:   (N_PIXELS,) numpy array — intensity frame from previous call

        Returns:
            ev_jax:        (1, N_PIXELS) JAX array  — event frame (batched for SLAM)
            kin_jax:       (1, 3) JAX array          — [vx_slam, vz_slam, w_theta]
            tof_jax:       (1, 3) JAX array          — 3-beam ToF distances (m, SLAM space)
            intensities:   (N_PIXELS,) numpy array   — current frame (store for next call)
        """
        # 1. Map hornet (x, z) → 10m SLAM coordinate frame
        slam_pos = jnp.array([
            float(robot_state_single[0]) * self._slam_scale + _SLAM_OFFSET,
            float(robot_state_single[1]) * self._slam_scale + _SLAM_OFFSET,
        ])
        heading = float(robot_state_single[2])
        # Camera and ToF sensors are mounted facing forward.
        # Since the nominal hover pitch of the hornet is 1.0 rad (tilted back/upright),
        # we subtract 1.0 rad from the body pitch to get the forward-facing camera/sensor angle.
        sensor_heading = heading - 1.0

        # 2. Event camera (256 pixels, 90° FOV)
        intensities_jax, _, _, _ = compute_pixel_readings(
            slam_pos, sensor_heading, self._segments,
            obstacles=self._obstacles, tex_tensor=self._tex_tensor,
        )
        intensities = np.array(intensities_jax)
        delta       = intensities - prev_intensities
        events      = np.where(delta >  THRESHOLD,  1.0,
                      np.where(delta < -THRESHOLD, -1.0, 0.0)).astype(np.float32)

        # 3. ToF (3-beam, values in SLAM metres; SLAM handles range internally)
        tof_jax = compute_tof_distance(slam_pos, sensor_heading, self._segments)

        # 4. Kinematic odometry [vx, vz, w_theta] (physical hornet units — m/s, rad/s)
        kin = np.array(robot_state_single[4:7], dtype=np.float32)

        # Return batched (B=1) JAX arrays matching SNNSLAMSystem.forward_step() signature
        ev_jax  = jnp.array(events[None, :])          # (1, N_PIXELS)
        kin_jax = jnp.array(kin[None, :])              # (1, 3)
        tof_out = jnp.array(tof_jax[None, :])          # (1, 3)

        return ev_jax, kin_jax, tof_out, intensities

    def slam_pose_to_hornet(self, slam_pose_xy: np.ndarray) -> np.ndarray:
        """
        Converts a SLAM pose (x, y) in 10m space back to hornet physical metres.
        Useful for feeding a corrected position into the Instar routing.

        Args:
            slam_pose_xy: (2,) or (B, 2) array of [slam_x, slam_y]
        Returns:
            hornet_xy: same shape, in hornet physical metres
        """
        return (np.asarray(slam_pose_xy) - _SLAM_OFFSET) / self._slam_scale
