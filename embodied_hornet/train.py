import os
import sys
import platform
# On Darwin Arm64 (macOS Apple Silicon), force CPU as Metal is unsupported.
# Otherwise, force CPU unless --gpu is passed.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif "--gpu" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import time

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import pickle
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from typing import NamedTuple
from functools import partial

# --- INTEGRATION MODULES (modified for embodied_hornet) ---
from .neural_idapbc import policy_network_icnn, differentiable_attention_gate, unpack_action, ScaleConfig, compute_sog_repulsive_force
from .reichardt_emd import compute_emd_intensities, compute_emd_signals, N_EMD_PIX
from .env import FlyEnv

# --- BASE MODULES FROM hornetRL SIBLING REPO ---
from hornetRL.fluid_surrogate import JaxSurrogateEngine
from hornetRL.fly_system import FlappingFlySystem, PhysParams
from hornetRL.neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from hornetRL.pbt_manager import init_pbt_state, pbt_evolve

# --- SLAM SYSTEM FROM neuro-symbolic-slam SUBMODULE ---
# (sys.path for src/ is configured in embodied_hornet/__init__.py)
from snn_slam_system import SNNSLAMSystem, N_DEPTH, SpikingOccupancyGrid
from sparse_forest import N_PIXELS, compute_tof_distance, compute_pixel_readings, THRESHOLD

# ---------------------------------------------------------------------------
# SOG Ray-Casting Helper (adapted from snn_live_slam.py)
# ---------------------------------------------------------------------------
def _get_ray_indices(cx, cy, cth, tof_dists, tof_angles, res, grid_size, offset_m=0.0, max_rays=300):
    """Ray-cast 3 ToF beams into hit/free grid indices for SOG LIF update.
    
    All coordinates are already in SLAM space (0–10m).
    offset_m: shift applied before converting to grid index (0 since we work in 0-10m directly).
    """
    hit_idx, free_idx = [], []
    MAX_VALID_RANGE = 2.83  # max diagonal of 2m×2m room = √(2²+2²)

    for i in range(3):
        d = float(tof_dists[i])
        trace_dist = min(d, MAX_VALID_RANGE)
        # Free-space cells along the beam
        for s in range(1, max(1, int(trace_dist / res))):
            fx = cx + (s * res) * np.cos(cth + tof_angles[i])
            fy = cy + (s * res) * np.sin(cth + tof_angles[i])
            fix = int((fx + offset_m) / res)
            fiy = int((fy + offset_m) / res)
            if 0 <= fix < grid_size and 0 <= fiy < grid_size:
                free_idx.append([fix, fiy])
        # Hit cell at beam end
        if d < MAX_VALID_RANGE:
            hx = cx + d * np.cos(cth + tof_angles[i])
            hy = cy + d * np.sin(cth + tof_angles[i])
            ix = int((hx + offset_m) / res)
            iy = int((hy + offset_m) / res)
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                hit_idx.append([ix, iy])

    # Pad to static shape for JAX JIT
    hit_pad  = np.full((max_rays, 2), -1, dtype=np.int32)
    free_pad = np.full((max_rays, 2), -1, dtype=np.int32)
    n_hit  = min(len(hit_idx),  max_rays)
    n_free = min(len(free_idx), max_rays)
    if n_hit  > 0: hit_pad[:n_hit]   = np.array(hit_idx)[:n_hit]
    if n_free > 0: free_pad[:n_free] = np.array(free_idx)[:n_free]
    return hit_pad, free_pad

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """
    Hyperparameters and physical constraints for the training pipeline.
    """
    SEED = 42

    # --- Master Frequency Setting ---
    BASE_FREQ = 115.0  
    
    # Target State: [x, z, theta, phi, vx, vz, w_theta, w_phi]
    TARGET_STATE = jnp.array([0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0])

    # --- Arena Boundaries ---
    # 2m × 2m physical arena (SLAM coord = physical coord + 1.0 → [0,2]m, slam_scale=1×)
    ARENA_W = 1.0

    # --- Time Scales ---
    DT = 3e-5               # Physics integration timestep (s)
    SIM_SUBSTEPS = 72       # Physics steps per Control Step
                            
    HORIZON = 32            # Trajectory horizon for Backpropagation (= 8 wingbeats @ 4 steps/beat)
    RESET_INTERVAL = 300     
    PBT_INTERVAL = 500      

    BATCH_SIZE = 32          
    LR_ACTOR = 5e-4
    LR_WARMUP_STEPS = 50        # ramp LR from LR_ACTOR/50 → LR_ACTOR over this many steps
    MAX_GRAD_NORM = 1.0
    GAMMA = 0.99
    DNAG_MIN_ALPHA = 0.3        # hover specialist always contributes ≥30% of the blend
                                # prevents SHAC's random critic from destroying hover stability

    OBS_NOISE_SIGMA = 0.002  
    ACTION_NOISE_SIGMA = 0.2

    TOTAL_UPDATES = 100000
    VIS_INTERVAL = 100
    CURRICULUM_RATIO = 0.5
    
    # --- Obstacle Avoidance Gains ---
    K_REPEL = 0.005
    K_FLOW = 0.3         # HS centering reflex gain (Dorsal stream — pitch torque)
    K_LOOM = 0.1         # LGMD looming escape gain (Dorsal stream — forward deceleration)
    N_EMD_PIX = 32       # Coarse ommatidial array resolution (Dorsal stream)
    EMD_TAU_BASE = 0.05       # Base EMD delay time constant (seconds) at 0 speed
    EMD_TAU_SPEED_GAIN = 2.0  # Gain for speed-adaptive EMD delay reduction
    
    # --- PBT Hyperparameters ---
    PBT_BASE_WEIGHTS = jnp.array([
        600.0,    # Pos
        10.0,     # Th_Ang
        4.0,     # Ab_Ang
        0.1,     # Lin_Vel
        0.02,     # Ang_Vel
        0.5      # Eff
    ])
    
    PBT_PERTURB_FACTOR = 1.2       
    PBT_TRUNCATE_FRACTION = 0.2    

    CKPT_DIR = "checkpoints_shac"
    VIS_DIR = "checkpoints_shac"
    AUX_LOSS_WEIGHT = 1.0
    VIS_INTERVAL = 100

    # --- Obstacle Collision ---
    OBS_PENALTY_WEIGHT = 50.0   # reward penalty per SLAM unit of penetration
    OBS_SAFETY_SLAM    = 0.1    # safety buffer in SLAM units (10cm physical in 2m room)

    WARMUP_STEPS = 1        

    FORCE_NORMALIZER = ScaleConfig.CONTROL_SCALE

# --- Observation Scaling SYMLOG ---
def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(combined_state, action_noise=None, SOG_v_mem=None, K_repel=0.0, emd_signals=None, K_flow=0.0, K_loom=0.0, robot_pos_slam=None):
    """
    Defines the Actor-Critic architecture over the unified state space.
    
    combined_state: Shape (Batch, 12) -> [physical_state (8), weighted_belief (4)]
    """
    # 1. Split observation vector into Physical State and Instar Perceptual belief
    physical_state = combined_state[..., :8]
    weighted_belief = combined_state[..., 8:]
    
    # 2. Prepare Target in SymLog Space
    target_sym = symlog(Config.TARGET_STATE)

    # 3. Actor (Brain + Muscles)
    mods, forces = policy_network_icnn(
        physical_state, 
        target_state=target_sym,
        action_noise=action_noise,
        SOG_v_mem=SOG_v_mem,
        K_repel=K_repel,
        emd_signals=emd_signals,
        K_flow=K_flow,
        K_loom=K_loom,
        robot_pos_slam=robot_pos_slam,
        dynamic_gains=True
    )
    
    # 4. Critic evaluates the unified 12-dimensional observation
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(combined_state)
    
    return mods, forces, value

ac_model = hk.without_apply_rng(hk.transform(actor_critic_fn))

# ==============================================================================
# 2b. HOVER SPECIALIST MODEL (8D — matches hover_params.pkl from hornetRL)
# ==============================================================================
def hover_actor_fn(physical_state, action_noise=None, SOG_v_mem=None, K_repel=0.0, emd_signals=None, K_flow=0.0, K_loom=0.0, robot_pos_slam=None):
    """
    Original hornetRL actor architecture (8D physical state only).
    Matches the parameter structure of hover_params.pkl — trained by hornetRL
    exclusively for stable hovering. Used as the fixed DNAG reference.
    NOT updated during SHAC training (hover_fixed_params is frozen).
    """
    target_sym = symlog(Config.TARGET_STATE)
    mods, forces = policy_network_icnn(
        physical_state,
        target_state=target_sym,
        action_noise=action_noise,
        SOG_v_mem=SOG_v_mem,
        K_repel=K_repel,
        emd_signals=emd_signals,
        K_flow=K_flow,
        K_loom=K_loom,
        robot_pos_slam=robot_pos_slam,
        dynamic_gains=False
    )
    # Critic on 8D state (matching hover_params.pkl structure)
    value = hk.Sequential([
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(128), jax.nn.tanh,
        hk.Linear(1)
    ])(physical_state)
    return mods, forces, value

hover_ac_model = hk.without_apply_rng(hk.transform(hover_actor_fn))

# ==============================================================================
# 3. VISUALIZATION ENGINE
# ==============================================================================
def run_visualization(env, params, update_idx, vis_step_fn, pbt_state=None, curr_state=None, slam_system=None, sog_state=None, target_xy=None, slam_pose=None, slam_surprise=0.0):
    print(f"--> Generating Visualization for Step {update_idx}...")
    import copy

    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 8

    sim_data = {
        'states': [], 'wing_pose': [], 'nodal_forces': [],
        'le_marker': [], 'hinge_marker': [], 't': [],
        'slam_pos': [],   # true physical position in SLAM space
        'slam_est': [],   # SLAM estimated position (world belief)
        'imu_dr': [],     # raw IMU dead-reckoning trajectory (no SLAM correction)
        'surprise': [],   # SLAM surprise metric
        'tof': [],        # (3,) ToF distances per frame (SLAM metres)
        'heading': [],    # sensor heading (rad) per frame
        'events': [],     # (256,) 1D event camera frames
        'active_places': [], # active place cell indices per frame
        'alpha': [],
        'f_repel': [],
        'flow_corr': [],
    }
    
    physics_dt = Config.DT * Config.SIM_SUBSTEPS * steps_per_frame
    kin_acc = np.zeros(3, dtype=np.float32)
    kin_count = 0
    
    best_idx = 0
    if pbt_state is not None:
        best_idx = int(np.argmax(pbt_state.running_reward))
        print(f"    -> Visualizing Best Agent (index {best_idx}) with reward {pbt_state.running_reward[best_idx]:.2f}")
    
    # Extract the best agent's parameters
    params_single = jax.tree.map(lambda x: x[best_idx], params)
    
    # Extract the best agent's state or reset if not provided
    if curr_state is not None:
        state = jax.tree.map(lambda x: x[best_idx : best_idx + 1], curr_state)
        print(f"    -> Resuming from exploration history (spawn pose: {state[0][0, :3]})")
    else:
        rng = jax.random.PRNGKey(update_idx)
        state = env.reset(rng, 1)
        r_state_override = state[0].at[0].set(Config.TARGET_STATE)
        state = (r_state_override,) + state[1:]

    # Copy the SLAM system if provided, otherwise create a new one
    if slam_system is not None:
        vis_slam = copy.deepcopy(slam_system)
        if best_idx != 0:
            # We must reset and initialize the pose tracker to match the best agent's spawn position
            r_state_np_start = np.array(state[0][0])
            start_slam_x = float(r_state_np_start[0]) * env._slam_scale + 1.0
            start_slam_z = float(r_state_np_start[1]) * env._slam_scale + 1.0
            start_slam_th = float(r_state_np_start[2]) - 1.0  # sensor heading
            
            vis_slam.reset_pose_only(1)
            vis_slam.initialize_pose(
                jnp.array([[start_slam_x, start_slam_z]]),
                jnp.array([start_slam_th]),
            )
            last_slam_est_u = start_slam_x
            last_slam_est_v = start_slam_z
            last_slam_est_th = start_slam_th
            print(f"    -> Deepcopied SLAM system but reset & re-initialized pose to best agent ({best_idx}) start: ({last_slam_est_u:.2f}, {last_slam_est_v:.2f})")
        else:
            last_slam_est_u = float(slam_pose[0])
            last_slam_est_v = float(slam_pose[1])
            last_slam_est_th = float(slam_pose[2])
            print(f"    -> Deepcopied SLAM system from training run at pose ({last_slam_est_u:.2f}, {last_slam_est_v:.2f})")
    else:
        from snn_slam_system import SNNSLAMSystem
        vis_slam = SNNSLAMSystem(jax.random.PRNGKey(update_idx + 999), n_depth=N_DEPTH)
        vis_slam.reset(1)
        
        r_state_np_start = np.array(state[0][0])
        start_slam_x = float(r_state_np_start[0]) * env._slam_scale + 1.0
        start_slam_z = float(r_state_np_start[1]) * env._slam_scale + 1.0
        start_slam_th = float(r_state_np_start[2]) - 1.0  # sensor heading
        env._prev_robot_state = None
        vis_slam.initialize_pose(
            jnp.array([[start_slam_x, start_slam_z]]),
            jnp.array([start_slam_th]),
        )
        last_slam_est_u = start_slam_x
        last_slam_est_v = start_slam_z
        last_slam_est_th = start_slam_th

    dr_x = last_slam_est_u
    dr_z = last_slam_est_v
    dr_th = last_slam_est_th
    raw_imu_x = last_slam_est_u
    raw_imu_z = last_slam_est_v
    raw_imu_th = last_slam_est_th
    slam_est_u = last_slam_est_u
    slam_est_v = last_slam_est_v
    slam_est_th = last_slam_est_th

    last_active_places = np.array([], dtype=np.int32)
    last_slam_surprise = float(slam_surprise) if slam_surprise is not None else 0.0
    slam_vis_csnn_jax = jnp.zeros((1, 256))
    slam_vis_stdp_jax = jnp.zeros((1, 256))
    
    active_props_batch = state[3] 
    real_props = jax.tree.map(lambda x: x[0], active_props_batch)

    # Load hover specialist for stabilizing visual rollout via DNAG blending
    _hover_pkl = os.path.join(os.path.dirname(__file__), "hover_params.pkl")
    hover_fixed_params = None
    if os.path.exists(_hover_pkl):
        with open(_hover_pkl, "rb") as _f:
            _hover_ckpt = pickle.load(_f)
        hover_fixed_params = jax.tree.map(jnp.array, _hover_ckpt['params'])
    hover_params_single = jax.tree.map(lambda x: x[0], hover_fixed_params) if hover_fixed_params is not None else None
    
    slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)
    vis_prev_int = np.zeros(N_PIXELS, dtype=np.float32)  # separate buffer for high-freq DVS display
    ev_jax = jnp.zeros((1, 256))

    # Real Spiking Occupancy Grid
    from snn_slam_system import SpikingOccupancyGrid
    _sog = SpikingOccupancyGrid(map_size_m=2.0, res=0.04, offset_m=0.0, v_max=1.0)
    if sog_state is not None:
        _sog_state = sog_state
        print("    -> Loaded SOG state from training run.")
    else:
        _sog_state = _sog.init_state()

    _TOF_ANGLES = [-np.pi/4, 0.0, np.pi/4]
    sim_data['sog_states'] = []  # per-frame v_mem snapshots for animation replay

    # Target waypoint: if target_xy is provided, use the best agent's active target
    if target_xy is not None:
        vis_target_xy = target_xy[best_idx : best_idx + 1]
        print(f"    -> Using best agent's target waypoint: {vis_target_xy}")
    else:
        vis_target_xy = jnp.array([[0.5, -0.5]])

    current_step_counter = 0
    vis_emd_intensities = jnp.zeros((1, Config.N_EMD_PIX))

    for i in range(total_visual_frames):
        r_st_start = np.array(state[0][0])
        for _ in range(steps_per_frame):
            r_st = state[0]
            r_cpu = np.array(r_st[0])
            if np.isnan(r_cpu).any():
                print(f"!!! Visualization stopped early due to NaN !!!")
                break
            
            slam_pose_jax = jnp.array([[slam_est_u, slam_est_v, slam_est_th]])
            vis_step_idx = current_step_counter + Config.WARMUP_STEPS + 5
            state, f_nodal, w_pose, h_marker, alpha_floored_jax, f_repel_scaled_jax, flow_corr_jax, vis_emd_intensities = vis_step_fn(
                env, state, params_single, vis_step_idx, jnp.array([last_slam_surprise]), hover_params_single, slam_pose_jax, slam_vis_csnn_jax, slam_vis_stdp_jax, _sog_state.v_mem, Config.K_REPEL, Config.K_FLOW, vis_target_xy, vis_emd_intensities
            )
            current_step_counter += 1

        r_state_np = np.array(state[0][0])  # (8,) physical state of agent 0
        
        # --- Check for early termination (crashes/OOR) to match closed-room physics ---
        is_oor = (abs(r_state_np[0]) > Config.ARENA_W) or (abs(r_state_np[1]) > Config.ARENA_W)
        _obs_np = np.array(env._obstacles) if env._obstacles is not None else np.zeros((0, 4))
        slam_u = r_state_np[0] * env._slam_scale + 1.0
        slam_v = r_state_np[1] * env._slam_scale + 1.0
        def _in_any_obs(sx, sz):
            if len(_obs_np) == 0: return False
            return bool(np.any(
                (_obs_np[:, 0] <= sx) & (_obs_np[:, 2] >= sx) &
                (_obs_np[:, 1] <= sz) & (_obs_np[:, 3] >= sz)
            ))
        is_in_obs = _in_any_obs(slam_u, slam_v)
        
        if is_oor or is_in_obs:
            print(f"!!! Visualization rollout stopped early due to {'boundary crash' if is_oor else 'obstacle collision'} !!!")
            break

        sim_data['states'].append(r_state_np)
        sim_data['t'].append(current_step_counter * Config.DT)
        # Convert physical (x, z) → SLAM (u, v) for nav panel
        _slam_scale  = env._slam_scale
        _slam_offset = 1.0
        sim_data['slam_pos'].append((slam_u, slam_v))
        
        # Camera/sensor heading is body pitch - 1.0 rad (facing forward)
        sensor_heading = float(r_state_np[2]) - 1.0
        
        # ToF: 3 beam distances from current SLAM position + sensor heading
        _tof_jax = compute_tof_distance(
            jnp.array([slam_u, slam_v]), sensor_heading, env._segments
        )
        sim_data['tof'].append(np.array(_tof_jax))
        sim_data['heading'].append(sensor_heading)

        # Accumulate kinematics at high frequency (115Hz) to prevent aliasing
        cos_sh = np.cos(sensor_heading)
        sin_sh = np.sin(sensor_heading)
        vx_sensor = (r_state_np[4] * cos_sh + r_state_np[5] * sin_sh) * env._slam_scale
        vz_sensor = (-r_state_np[4] * sin_sh + r_state_np[5] * cos_sh) * env._slam_scale
        w_theta = r_state_np[6]
        kin_acc += np.array([vx_sensor, vz_sensor, w_theta], dtype=np.float32)
        kin_count += 1

        # Capture the starting state at the beginning of the 10-step block
        if i % 10 == 0:
            r_st_start_10 = np.array(r_st_start)

        # Dead-reckoning position integrator (high frequency)
        dr_vx = float(r_state_np[4]) * env._slam_scale
        dr_vz = float(r_state_np[5]) * env._slam_scale
        dr_vth = float(r_state_np[6]) # angular velocity (yaw rate)
        
        # Integrate raw IMU dead-reckoning (without feedback correction)
        raw_imu_x += dr_vx * physics_dt
        raw_imu_z += dr_vz * physics_dt
        raw_imu_th += dr_vth * physics_dt
        raw_imu_th = (raw_imu_th + np.pi) % (2 * np.pi) - np.pi
        sim_data['imu_dr'].append((raw_imu_x, raw_imu_z, raw_imu_th))
        
        # Proportional feedback correction towards CANN estimate (prevents steps/discontinuities)
        K_CORR = 20.0
        err_x = last_slam_est_u - dr_x
        err_z = last_slam_est_v - dr_z
        err_th = last_slam_est_th - dr_th
        err_th = (err_th + np.pi) % (2 * np.pi) - np.pi
        
        dr_x += (dr_vx + K_CORR * err_x) * physics_dt
        dr_z += (dr_vz + K_CORR * err_z) * physics_dt
        dr_th += (dr_vth + K_CORR * err_th) * physics_dt
        dr_th = (dr_th + np.pi) % (2 * np.pi) - np.pi
        
        slam_est_u = dr_x
        slam_est_v = dr_z
        slam_est_th = dr_th

        # Trigger CANN update exactly every 10 steps to align time-scales and eliminate timing jitter
        if (i + 1) % 10 == 0:
            elapsed_time = 10 * physics_dt
            
            # SLAM tracking for visualization - run once per 10 steps (50Hz)
            env._prev_robot_state = r_st_start_10
            ev_jax, _, tof_jax, acc_jax, slam_prev_int = env.compute_slam_sensors(
                r_state_np, slam_prev_int, dt=elapsed_time
            )
            
            avg_kin = kin_acc / max(1, kin_count)
            
            try:
                pose_est, _, _, _, _, debug_gates = vis_slam.forward_step(
                    ev_jax, jnp.array([avg_kin]), tof_jax,
                    acc_t=acc_jax,
                    inject_drift=False, autopilot_on=(last_slam_surprise < 0.60),
                    dt=elapsed_time
                )
                last_slam_surprise = float(1.0 - float(debug_gates['Raw_Match'][0]))
                
                # Use actual CANN SLAM output for position and heading display
                last_slam_est_u = float(pose_est[0, 0])
                last_slam_est_v = float(pose_est[0, 1])
                last_slam_est_th = float(pose_est[0, 2])
                
                # Print professional telemetry tracking
                est_pitch = float(pose_est[0, 2])
                est_grav = float(vis_slam._theta_gravity[0])
                print(f"  📊 [SLAM TELEMETRY] Step {i:03d} | Pos GT: ({slam_u:.2f}, {slam_v:.2f}) SLAM: ({last_slam_est_u:.2f}, {last_slam_est_v:.2f}) "
                      f"| Pitch GT: {r_state_np[2] - 1.0:.3f} rad, CF Grav: {est_grav:.3f} rad, CANN Head: {est_pitch:.3f} rad | Surprise: {last_slam_surprise:.3f}")
                
                slam_vis_csnn_jax = jnp.array(debug_gates['Debug_Input_CSNN'])
                slam_vis_stdp_jax = jnp.array(debug_gates['Debug_Input_STDP'])
                
                I_place_np = np.array(debug_gates['Debug_I_Place'][0])
                max_val = np.max(I_place_np)
                last_active_places = np.where(I_place_np > 0.1 * max_val)[0] if max_val > 1e-5 else np.array([], dtype=np.int32)
                
            except Exception as _slam_err:
                print(f"!!! SLAM error at frame {current_step_counter}: {_slam_err}")
                last_slam_surprise = 0.0
                last_active_places = np.array([], dtype=np.int32)
            
            # Reset accumulators
            kin_acc = np.zeros(3, dtype=np.float32)
            kin_count = 0

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  🎬 [VIS FRAME {i:03d}/{total_visual_frames:03d}] Rendering frame and updating physics...")

        sim_data['active_places'].append(last_active_places)
        sim_data['slam_est'].append((slam_est_u, slam_est_v, slam_est_th))
        sim_data['surprise'].append(last_slam_surprise)
        # Compute events at every frame (463 Hz) for smooth DVS display
        _vis_slam_pos = jnp.array([slam_u, slam_v])
        _vis_int, _, _, _ = compute_pixel_readings(
            _vis_slam_pos, sensor_heading, env._segments,
            obstacles=env._obstacles, tex_tensor=env._tex_tensor,
        )
        _vis_int_np = np.array(_vis_int)
        _vis_delta = _vis_int_np - vis_prev_int
        _vis_events = np.where(_vis_delta > THRESHOLD, 1.0,
                     np.where(_vis_delta < -THRESHOLD, -1.0, 0.0)).astype(np.float32)
        vis_prev_int = _vis_int_np
        sim_data['events'].append(_vis_events)
        sim_data['alpha'].append(float(alpha_floored_jax.squeeze()))
        sim_data['f_repel'].append(float(jnp.linalg.norm(f_repel_scaled_jax.squeeze())))
        sim_data['flow_corr'].append(float(flow_corr_jax.squeeze()))

        # Real SOG update: ray-cast ToF beams, excite hits, inhibit free space
        tof_d = sim_data['tof'][-1]
        _hit_idx, _free_idx = _get_ray_indices(
            slam_est_u, slam_est_v, slam_est_th,
            tof_d, _TOF_ANGLES,
            res=_sog.res, grid_size=_sog.grid_w, offset_m=_sog.offset_m,
        )
        _sog_state = _sog.update(_sog_state, jnp.array(_hit_idx), jnp.array(_free_idx))
        # Snapshot the current v_mem for animation replay
        sim_data['sog_states'].append(np.array(_sog_state.v_mem))

        f_st = state[1]
        sim_data['le_marker'].append(np.array(f_st.marker_le[0]))
        sim_data['wing_pose'].append(np.array(w_pose[0]))
        sim_data['nodal_forces'].append(np.array(f_nodal[0]))
        sim_data['hinge_marker'].append(np.array(h_marker[0]))

    # Store final sequence weights and SOG metadata
    sim_data['final_W_seq'] = np.array(vis_slam.place_state.W_seq_to_place[0])
    sim_data['sog_res'] = _sog.res
    sim_data['sog_grid_w'] = _sog.grid_w

    # -----------------------------------------------------------------------
    # Matplotlib Animation — 2x3 Scientific Dashboard Grid
    # Col 0 (Rows 0-1): Navigation Room (ax_nav)
    # Col 1 (Row 0):    SOG Heatmap (ax_map)
    # Col 2 (Row 0):    Wing Mechanics (ax_wing)
    # Col 1 (Row 1):    Place Cell Spikes Raster (ax_snn)
    # Col 2 (Row 1):    Dual-Axis Telemetry Chart (ax_telemetry)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('#0d0d0d')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.0], width_ratios=[1.2, 1.0, 1.0])
    
    ax_nav = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_wing = fig.add_subplot(gs[0, 2])
    ax_snn = fig.add_subplot(gs[1, 1])
    ax_telemetry = fig.add_subplot(gs[1, 2])
    
    for ax in (ax_nav, ax_map, ax_wing, ax_snn, ax_telemetry):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # --- Navigation Room ---
    ax_nav.set_xlim(0, 2)
    ax_nav.set_ylim(0, 2)
    ax_nav.set_aspect('equal')
    ax_nav.set_title('Navigation Room (SLAM Space)', color='white', fontsize=10, fontweight='bold')
    ax_nav.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
    ax_nav.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)

    # Draw room boundary
    room_rect = plt.Rectangle((0, 0), 2, 2, linewidth=2, edgecolor='#00ffcc', facecolor='none')
    ax_nav.add_patch(room_rect)

    # Draw obstacles as dynamic patches (re-colored on collision)
    obstacles_np = np.array(env._obstacles) if env._obstacles is not None else np.zeros((0, 4))
    obs_patch_list = []
    for obs in obstacles_np:
        x0, y0, x1, y1 = obs
        p = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, zorder=4)
        ax_nav.add_patch(p)
        obs_patch_list.append((obs, p))

    # Draw target in SLAM coords using best agent's active target
    target_phys = np.array(vis_target_xy[0])
    _slam_scale = env._slam_scale
    _slam_offset = 1.0
    tgt_u = target_phys[0] * _slam_scale + _slam_offset
    tgt_v = target_phys[1] * _slam_scale + _slam_offset
    ax_nav.plot(tgt_u, tgt_v, marker='*', markersize=14, color='#ffdd00',
                markeredgecolor='#ff8800', zorder=20, label='Target')

    # Trajectory + artists
    traj_line, = ax_nav.plot([], [], '-', color='#00ff88', linewidth=1.0, alpha=0.6, zorder=5, label='Path (GT)')
    hornet_dot, = ax_nav.plot([], [], 'o', color='#ff4444', markersize=7, zorder=15, label='Hornet (GT)')
    slam_est_line, = ax_nav.plot([], [], ':', color='#ffa500', linewidth=1.2, alpha=0.8, zorder=6, label='Path (SLAM)')
    slam_est_dot, = ax_nav.plot([], [], 's', color='#ffa500', markersize=5, zorder=14, label='SLAM Pose')
    imu_dr_line, = ax_nav.plot([], [], '--', color='#00ffff', linewidth=0.8, alpha=0.5, zorder=4, label='Path (IMU DR)')
    imu_dr_dot, = ax_nav.plot([], [], '^', color='#00ffff', markersize=5, zorder=13, label='IMU DR Pose')
    heading_arr = ax_nav.quiver([0], [0], [0], [0], color='#ff8888', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Sensor Heading')
    slam_heading_arr = ax_nav.quiver([0], [0], [0], [0], color='#ffa500', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=15, label='SLAM Heading')
    vel_arr = ax_nav.quiver([0], [0], [0], [0], color='#ff33ff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Velocity')

    # 3 ToF beam artists
    _beam_colours = ['#00aaff', '#ffff44', '#00aaff']
    tof_beam_artists = []
    for _bc in _beam_colours:
        _bl, = ax_nav.plot([], [], '-',  color=_bc, linewidth=1.5, alpha=0.85, zorder=12)
        _bm, = ax_nav.plot([], [], 'D',  color=_bc, markersize=4,  alpha=0.95, zorder=13)
        tof_beam_artists.append((_bl, _bm))

    # Camera FOV boundary dashes
    fov_left_line,  = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
    fov_right_line, = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)

    nav_time = ax_nav.text(0.02, 0.97, '', transform=ax_nav.transAxes,
                           color='#cccccc', fontsize=8, va='top', family='monospace')
    ax_nav.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)

    # 1D Event Camera Vision Strip (Inset axis at the top of the nav view)
    from matplotlib.colors import LinearSegmentedColormap
    ax_vis = ax_nav.inset_axes([0.05, 0.91, 0.9, 0.025])
    ax_vis.set_xticks([])
    ax_vis.set_yticks([])
    for spine in ax_vis.spines.values():
        spine.set_edgecolor('#444444')
    ax_vis.set_title('1D Event Camera Stream (Green=ON, Red=OFF)', color='#888888', fontsize=6, pad=1)
    
    colors_ev = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.8, 0.1)]
    cm_ev = LinearSegmentedColormap.from_list('events_cmap', colors_ev, N=3)
    vis_strip = ax_vis.imshow(np.zeros((1, N_PIXELS)), cmap=cm_ev, vmin=-1.0, vmax=1.0, aspect='auto')

    # --- SOG Heatmap (Robot Memory) ---
    ax_map.set_aspect('equal')
    ax_map.set_title('Spiking Occupancy Grid (Robot Memory)', color='white', fontsize=10, fontweight='bold')
    ax_map.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
    ax_map.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)

    SOG_EXTENT = [0, 2, 0, 2]
    _grid_n = sim_data.get('sog_grid_w', 50)
    occ_display = np.zeros((_grid_n, _grid_n), dtype=np.float32)
    occ_img = ax_map.imshow(
        occ_display, origin='lower', extent=SOG_EXTENT,
        cmap='magma', vmin=-0.2, vmax=1.0, aspect='equal', zorder=1,
    )
    map_robot_dot, = ax_map.plot([], [], 'o', color='#00ffcc', markersize=5, zorder=10, label='Robot')
    map_trail_line, = ax_map.plot([], [], '-', color='#00ffcc', linewidth=0.8, alpha=0.4, zorder=5, label='Trail')

    map_tof_beam_artists = []
    for _bc in _beam_colours:
        _bl, = ax_map.plot([], [], '-',  color=_bc, linewidth=1.2, alpha=0.6, zorder=12)
        _bm, = ax_map.plot([], [], 'D',  color=_bc, markersize=3,  alpha=0.7, zorder=13)
        map_tof_beam_artists.append((_bl, _bm))

    map_fov_left_line,  = ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
    map_fov_right_line, = ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
    ax_map.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)

    # --- Wing Mechanics ---
    ax_wing.set_aspect('equal')
    ax_wing.set_title('Wing Mechanics (Close-up)', color='white', fontsize=10, fontweight='bold')
    ax_wing.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
    ax_wing.set_ylabel('Z (m)', color='#aaaaaa', fontsize=8)
    ax_wing.grid(True, linestyle=':', alpha=0.2, color='#444444')

    room_rect_wing = plt.Rectangle((-1.0, -1.0), 2.0, 2.0, linewidth=2, edgecolor='#00ffcc', facecolor='none', linestyle='--', zorder=2)
    ax_wing.add_patch(room_rect_wing)

    obs_patch_wing_list = []
    for obs in obstacles_np:
        px0, py0, px1, py1 = obs - 1.0
        p_wing = plt.Rectangle((px0, py0), px1 - px0, py1 - py0,
                               facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, alpha=0.8, zorder=3)
        ax_wing.add_patch(p_wing)
        obs_patch_wing_list.append((obs - 1.0, p_wing))

    patch_thorax = patches.Ellipse((0,0), linewidth=1.0, width=0.012, height=0.006,
                                   facecolor='#555555', edgecolor='#aaaaaa', zorder=10)
    patch_head   = patches.Circle((0,0), linewidth=1.0, radius=0.0025,
                                   facecolor='#00FF88', edgecolor='#aaaaaa', zorder=10)
    patch_abd    = patches.Ellipse((0,0), linewidth=1.0, width=0.018, height=0.008,
                                   facecolor='#4488cc', edgecolor='#aaaaaa', alpha=0.8, zorder=9)
    ax_wing.add_patch(patch_thorax)
    ax_wing.add_patch(patch_head)
    ax_wing.add_patch(patch_abd)

    real_line, = ax_wing.plot([], [], '-', color='#cccccc', linewidth=1.5, alpha=0.9, zorder=12)
    patch_le    = patches.Circle((0,0), radius=0.001, color='#ff4444', zorder=15)
    patch_hinge = patches.Circle((0,0), radius=0.001, color='#ff8800', zorder=15)
    ax_wing.add_patch(patch_le)
    ax_wing.add_patch(patch_hinge)

    dummy = np.zeros(20)
    quiver = ax_wing.quiver(dummy, dummy, dummy, dummy, color='#ff6666',
                             scale=3.0, scale_units='xy', zorder=20, width=0.0002)
    time_text = ax_wing.text(0.02, 0.95, '', transform=ax_wing.transAxes, color='#cccccc', fontsize=8)

    # --- Place Cell Spike Raster ---
    spike_t = []
    spike_idx = []
    for f, active in enumerate(sim_data['active_places']):
        t_val = sim_data['t'][f]
        for neuron_id in active:
            spike_t.append(t_val)
            spike_idx.append(neuron_id)

    ax_snn.scatter(spike_t, spike_idx, s=2, color='#00ffff', alpha=0.4, label='Place Cell Spikes')
    ax_snn.set_xlim(0, sim_data['t'][-1] if sim_data['t'] else 1.0)
    ax_snn.set_ylim(-5, 260)
    ax_snn.set_title('Place Cell Spike Raster', color='white', fontsize=10, fontweight='bold')
    ax_snn.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
    ax_snn.set_ylabel('Neuron ID (0-255)', color='#aaaaaa', fontsize=8)
    ax_snn.grid(True, linestyle=':', alpha=0.2, color='#444444')
    snn_time_line = ax_snn.axvline(x=0, color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)

    # --- Closed-Loop Telemetry Chart ---
    time_series = sim_data['t']
    line_surprise, = ax_telemetry.plot(time_series, sim_data['surprise'], '-', color='#ffa500', linewidth=1.2, label='SLAM Surprise')
    line_alpha, = ax_telemetry.plot(time_series, sim_data['alpha'], '-', color='#00ffff', linewidth=1.2, label='DNAG Alpha (Gate)')
    
    ax_telemetry.set_xlim(0, time_series[-1] if time_series else 1.0)
    ax_telemetry.set_ylim(-0.05, 1.1)
    ax_telemetry.set_title('Closed-Loop Telemetry', color='white', fontsize=10, fontweight='bold')
    ax_telemetry.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
    ax_telemetry.set_ylabel('Surprise / Alpha', color='#cccccc', fontsize=8)
    ax_telemetry.grid(True, linestyle=':', alpha=0.2, color='#444444')

    ax_telemetry_right = ax_telemetry.twinx()
    ax_telemetry_right.set_facecolor('none')
    ax_telemetry_right.tick_params(colors='#aaaaaa', labelsize=8)
    for spine in ax_telemetry_right.spines.values():
        spine.set_edgecolor('#444444')
        
    line_repel, = ax_telemetry_right.plot(time_series, sim_data['f_repel'], '-', color='#ff33ff', linewidth=1.2, label='SOG Repulsion')
    max_repel = max(sim_data['f_repel']) if sim_data['f_repel'] else 1.0
    ax_telemetry_right.set_ylim(-0.05 * max_repel, max_repel * 1.1 + 1e-3)
    ax_telemetry_right.set_ylabel('SOG Repulsion (N)', color='#ff33ff', fontsize=8)

    lines = [line_surprise, line_alpha, line_repel]
    labels = [l.get_label() for l in lines]
    ax_telemetry.legend(lines, labels, loc='upper left', facecolor='#222222', labelcolor='white', fontsize=7)
    
    telemetry_time_line = ax_telemetry.axvline(x=0, color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)

    def update(frame):
        if frame >= len(sim_data['states']): return

        r_state  = sim_data['states'][frame]
        w_pose   = sim_data['wing_pose'][frame]
        f_nodal  = sim_data['nodal_forces'][frame]
        le_pos   = sim_data['le_marker'][frame]
        hinge_p  = sim_data['hinge_marker'][frame]
        t        = sim_data['t'][frame]
        slam_pos = sim_data['slam_pos'][:frame+1]

        rx, rz   = r_state[0], r_state[1]
        r_th, r_phi = r_state[2], r_state[3]

        # -- left panel: SLAM nav view --
        if slam_pos:
            xs = [p[0] for p in slam_pos]
            ys = [p[1] for p in slam_pos]
            traj_line.set_data(xs, ys)
            hornet_dot.set_data([xs[-1]], [ys[-1]])
            
            # SLAM estimated path, pose dot, and heading arrow (world belief)
            if frame < len(sim_data['slam_est']):
                est_pts = sim_data['slam_est']
                est_xs = [p[0] for p in est_pts[:frame+1]]
                est_ys = [p[1] for p in est_pts[:frame+1]]
                est_th = est_pts[frame][2]
                slam_est_line.set_data(est_xs, est_ys)
                slam_est_dot.set_data([est_xs[-1]], [est_ys[-1]])
                slam_heading_arr.set_offsets([[est_xs[-1], est_ys[-1]]])
                slam_heading_arr.set_UVC([0.2 * np.cos(est_th)], [0.2 * np.sin(est_th)])
                
            # Raw IMU dead-reckoning path and pose dot
            if frame < len(sim_data['imu_dr']):
                imu_pts = sim_data['imu_dr']
                imu_xs = [p[0] for p in imu_pts[:frame+1]]
                imu_ys = [p[1] for p in imu_pts[:frame+1]]
                imu_dr_line.set_data(imu_xs, imu_ys)
                imu_dr_dot.set_data([imu_xs[-1]], [imu_ys[-1]])

            # === SLAM sensor layer ===
            if frame < len(sim_data['tof']):
                tof_d = sim_data['tof'][frame]
                hdg   = sim_data['heading'][frame]
                cu, cv = xs[-1], ys[-1]

                # Update heading arrow pointing in sensor heading
                heading_arr.set_offsets([[cu, cv]])
                heading_arr.set_UVC([0.2 * np.cos(hdg)], [0.2 * np.sin(hdg)])

                # Update velocity arrow in SLAM coordinates
                vx_phys = r_state[4]
                vz_phys = r_state[5]
                vx_slam = vx_phys * env._slam_scale
                vy_slam = vz_phys * env._slam_scale
                vel_arr.set_offsets([[cu, cv]])
                vel_arr.set_UVC([0.2 * vx_slam], [0.2 * vy_slam])

                # --- Update Occupancy Grid — replay SOG v_mem ---
                if frame < len(sim_data['sog_states']) and frame < len(sim_data['slam_est']):
                    est_u, est_v, est_th = sim_data['slam_est'][frame]
                    v_mem = sim_data['sog_states'][frame]
                    occ_img.set_data(v_mem.T)
                    occ_img.set_clim(vmin=-0.2, vmax=1.0)

                    # Update robot dot and trail on the map
                    map_robot_dot.set_data([est_u], [est_v])
                    trail_pts = sim_data['slam_est'][:frame+1]
                    map_trail_line.set_data(
                        [p[0] for p in trail_pts],
                        [p[1] for p in trail_pts],
                    )

                    # Update ToF beams on SOG map
                    for bi, ((bl, bm), offset) in enumerate(
                            zip(map_tof_beam_artists, [-np.pi/4, 0.0, np.pi/4])):
                        ang = est_th + offset
                        hu  = est_u + tof_d[bi] * np.cos(ang)
                        hv  = est_v + tof_d[bi] * np.sin(ang)
                        bl.set_data([est_u, hu], [est_v, hv])
                        bm.set_data([hu], [hv])

                    # Update FOV boundary on SOG map
                    fov_r = 1.2
                    map_fov_left_line.set_data(
                        [est_u, est_u + fov_r * np.cos(est_th - np.pi/4)],
                        [est_v, est_v + fov_r * np.sin(est_th - np.pi/4)])
                    map_fov_right_line.set_data(
                        [est_u, est_u + fov_r * np.cos(est_th + np.pi/4)],
                        [est_v, est_v + fov_r * np.sin(est_th + np.pi/4)])

                # 3 ToF beams: left/center/right at ±45° from heading
                for bi, ((bl, bm), offset) in enumerate(
                        zip(tof_beam_artists, [-np.pi/4, 0.0, np.pi/4])):
                    ang = hdg + offset
                    hu  = cu + tof_d[bi] * np.cos(ang)
                    hv  = cv + tof_d[bi] * np.sin(ang)
                    bl.set_data([cu, hu], [cv, hv])
                    bm.set_data([hu], [hv])

                # FOV boundary (90° camera cone)
                fov_r = 1.2
                fov_left_line.set_data(
                    [cu, cu + fov_r * np.cos(hdg - np.pi/4)],
                    [cv, cv + fov_r * np.sin(hdg - np.pi/4)])
                fov_right_line.set_data(
                    [cu, cu + fov_r * np.cos(hdg + np.pi/4)],
                    [cv, cv + fov_r * np.sin(hdg + np.pi/4)])

                # Obstacle collision highlighting
                for obs_bbox, op in obs_patch_list:
                    x0, y0, x1, y1 = obs_bbox
                    inside = (x0 <= cu <= x1) and (y0 <= cv <= y1)
                    op.set_facecolor('#cc1111' if inside else '#2a2a4c')
                    op.set_edgecolor('#ff3333' if inside else '#6666bb')

                # Update 1D event camera vision strip
                if frame < len(sim_data['events']):
                    ev_frame = sim_data['events'][frame]
                    vis_strip.set_data(ev_frame[None, :])

                # SLAM text overlay
                surprise_val = sim_data['surprise'][frame] if frame < len(sim_data['surprise']) else 0.0
                alpha_val = sim_data['alpha'][frame] if frame < len(sim_data['alpha']) else 0.0
                nav_time.set_text(
                    f'θ={hdg:.2f}r | ToF [{tof_d[0]:.1f}│{tof_d[1]:.1f}│{tof_d[2]:.1f}]m | Surprise={surprise_val:.2f} | α={alpha_val:.2f}'
                )
            else:
                nav_time.set_text(f'T={t:.3f}s')

        # -- right panel: wing mechanics (zoomed out to 30cm window for context) --
        ax_wing.set_xlim(rx - 0.15, rx + 0.15)
        ax_wing.set_ylim(rz - 0.15, rz + 0.15)

        # Highlight obstacles on collision in wing panel
        for obs_phys, op_wing in obs_patch_wing_list:
            px0, py0, px1, py1 = obs_phys
            inside = (px0 <= rx <= px1) and (py0 <= rz <= py1)
            op_wing.set_facecolor('#cc1111' if inside else '#2a2a4c')
            op_wing.set_edgecolor('#ff3333' if inside else '#6666bb')

        d1 = real_props.d1
        d2 = real_props.d2
        patch_thorax.set_center((rx, rz))
        patch_thorax.set_angle(np.degrees(r_th))
        patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
        joint_x = rx - d1 * np.cos(r_th)
        joint_z = rz - d1 * np.sin(r_th)
        abd_ang = r_th + r_phi
        patch_abd.set_center((joint_x - d2 * np.cos(abd_ang), joint_z - d2 * np.sin(abd_ang)))
        patch_abd.set_angle(np.degrees(abd_ang))

        wx, wz, wang = w_pose
        wing_len = env.phys.fluid.WING_LEN
        N_pts    = env.phys.fluid.N_PTS
        x_local  = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        wing_x   = wx + x_local * c_w
        wing_z   = wz + x_local * s_w
        real_line.set_data(wing_x, wing_z)
        patch_le.set_center((wing_x[0], wing_z[0]))
        patch_hinge.set_center((hinge_p[0], hinge_p[1]))
        pts = np.stack([wing_x, wing_z], axis=1)
        quiver.set_offsets(pts)
        quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
        time_text.set_text(f'T={t:.4f}s | Z={rz:.3f}m')

        # -- time indicator lines --
        snn_time_line.set_xdata([t])
        telemetry_time_line.set_xdata([t])

        return (patch_thorax, patch_le, patch_hinge, traj_line, hornet_dot, slam_est_line,
                slam_est_dot, imu_dr_line, imu_dr_dot, heading_arr, slam_heading_arr,
                snn_time_line, telemetry_time_line)

    plt.tight_layout()
    if len(sim_data['states']) == 0:
        print(f"--> Viz skipped: rollout ended before any frames were recorded.")
        plt.close(fig)
        return
    ani = animation.FuncAnimation(fig, update, frames=len(sim_data['states']), interval=20, blit=False)
    out_file = os.path.join(Config.VIS_DIR, f"epoch_{update_idx}.gif")
    ani.save(out_file, writer='pillow', fps=60)
    plt.close(fig)
    print(f"--> Saved Viz: {out_file}")

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def train():
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    os.makedirs(Config.VIS_DIR, exist_ok=True)
    
    env = FlyEnv(Config)
    rng = jax.random.PRNGKey(Config.SEED)
    
    # 🌟 Dummy input updated to 12 dimensions to hold combined physical & perceptual observations
    dummy_input = jnp.zeros((1, 12))

    checkpoints = glob.glob(os.path.join(Config.CKPT_DIR, "*.pkl"))
    hornet_path = "hornet_brain.pkl" 
    
    start_step = 0
    
    if checkpoints:
        checkpoints.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        last_ckpt = checkpoints[-1]
        print(f"--> [RESUME] Found checkpoint: {last_ckpt}")
        
        with open(last_ckpt, "rb") as f:
            data = pickle.load(f)
        
        params = data['params'] 
        opt_state = data['opt_state']
        
        if 'pbt_state' in data:
            pbt_state = data['pbt_state']
            print("    -> PBT State loaded.")
        else:
            pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)
            print("    -> WARNING: No PBT state found. Resetting PBT curriculum.")

        match = re.search(r"params_(\d+)", last_ckpt)
        if match: start_step = int(match.group(1)) + 1
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )

    elif os.path.exists(hornet_path):
        print(f"--> [TRANSFER] No checkpoint found. Loading Expert: {hornet_path}")
        with open(hornet_path, "rb") as f:
            expert_data = pickle.load(f)
            
        single_params = expert_data['params']
        params = jax.tree.map(lambda x: jnp.stack([x] * Config.BATCH_SIZE), single_params)
        
        _lr_schedule = optax.linear_schedule(
            init_value=Config.LR_ACTOR / 50.0,
            end_value=Config.LR_ACTOR,
            transition_steps=Config.LR_WARMUP_STEPS,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(_lr_schedule)
        )
        opt_state = optimizer.init(params)
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    else:
        print("--> [SCRATCH] No checkpoint or expert found. Initializing random population.")
        single_params = ac_model.init(rng, dummy_input)
        params = jax.tree.map(lambda x: jnp.stack([x] * Config.BATCH_SIZE), single_params)

        # --- Warm-start ICNN + BiologicalKinematicMap from hover_params.pkl ---
        # hover_params.pkl was trained to produce stable hovering. Copying its
        # ICNN and kinematic map weights into the SHAC policy prevents NaN at
        # step 0: random ICNN weights produce extreme gradients → huge forces →
        # physics diverges instantly.  Only the value-head (linear/linear_1/
        # linear_2) is kept random because its first-layer input dim is 12 here
        # vs 8 there (not compatible).
        #
        # We use the FULL 32-agent PBT batch directly — each SHAC agent gets a
        # different trained hover specialist's weights, providing diversity while
        # ensuring all start from a numerically stable regime.
        _hover_pkl_ws = os.path.join(os.path.dirname(__file__), "hover_params.pkl")
        if os.path.exists(_hover_pkl_ws):
            with open(_hover_pkl_ws, "rb") as _fws:
                _hover_ws = pickle.load(_fws)
            _hover_full = jax.tree.map(jnp.array, _hover_ws['params'])
            _value_head_keys = {'linear', 'linear_1', 'linear_2'}  # skip: 8→128 ≠ 12→128
            _copied = []
            for k in _hover_full:
                if k not in _value_head_keys and k in params:
                    params[k] = _hover_full[k]  # (32,...) — full diverse PBT population
                    _copied.append(k)
            print(f"--> [WARM-START] Copied {len(_copied)} param groups from hover_params.pkl "
                  f"(full 32-agent PBT diversity): {_copied}")
        else:
            print("--> [WARM-START] hover_params.pkl not found — keeping random init (expect NaN at step 0)")

        _lr_schedule = optax.linear_schedule(
            init_value=Config.LR_ACTOR / 50.0,
            end_value=Config.LR_ACTOR,
            transition_steps=Config.LR_WARMUP_STEPS,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(_lr_schedule)
        )
        opt_state = optimizer.init(params)
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    print(f"--> Initialization Complete. Params Batch Shape: {params['linear']['w'].shape}")

    # --------------------------------------------------------------------------
    # Load hover specialist (fixed reference for DNAG gate)
    # --------------------------------------------------------------------------
    _hover_pkl = os.path.join(os.path.dirname(__file__), "hover_params.pkl")
    _use_hover_specialist = False
    hover_fixed_params = None

    if os.path.exists(_hover_pkl):
        with open(_hover_pkl, "rb") as _f:
            _hover_ckpt = pickle.load(_f)
        hover_fixed_params = jax.tree.map(jnp.array, _hover_ckpt['params'])
        _hover_batch = jax.tree.leaves(hover_fixed_params)[0].shape[0]
        _use_hover_specialist = True
        print(f"--> [HOVER] Loaded hover specialist from hover_params.pkl "
              f"(batch={_hover_batch}, keys={list(hover_fixed_params.keys())})")
    else:
        print("--> [HOVER] hover_params.pkl not found — using velocity-zeroed policy fallback.")

    # Batched hover network (vmapped over 32-agent PBT population)
    batched_hover_network = jax.vmap(
        hover_ac_model.apply,
        in_axes=(0, 0, 0, None, None, 0, None, None, 0)
    )

    # Validate hover specialist with a dry-run forward pass before JIT
    if _use_hover_specialist:
        try:
            _dummy_8d   = jnp.zeros((_hover_batch, 8))
            _dummy_act  = jnp.zeros((_hover_batch, 4))
            _dummy_sog  = jnp.zeros((50, 50))
            _dummy_emd  = jnp.zeros((_hover_batch, 2))
            _dummy_pos  = jnp.zeros((_hover_batch, 2))
            _test_mods, _, _ = batched_hover_network(
                hover_fixed_params, _dummy_8d, _dummy_act, _dummy_sog, 0.0, _dummy_emd, 0.0, 0.0, _dummy_pos
            )
            print(f"    -> Hover specialist validated: mods shape {_test_mods.shape}")
        except Exception as _e:
            print(f"    -> WARNING: hover specialist forward-pass failed ({_e})")
            print(f"       Param keys: {list(hover_fixed_params.keys())}")
            print(f"       Falling back to velocity-zeroed policy.")
            _use_hover_specialist = False
            hover_fixed_params    = None

    def loss_fn(params, start_state, pbt_weights, key, slam_pose, slam_surprise, obstacles, vis_csnn, vis_stdp, SOG_v_mem, target_xy):
        """
        Computes the total loss over the trajectory horizon.
        Includes policy gradient, value function loss, and auxiliary force matching loss.

        slam_pose:     (3,) JAX array  [slam_x, slam_y, slam_heading] in 10m SLAM space.
                       Converted to hornet metres before being fed to the Instar routing.
        slam_surprise: scalar float  in [0, 1]  — 1.0 = fully novel scene, 0.0 = familiar.
        target_xy:     (B, 2) JAX array of dynamic target physical coordinates.
        """
        rollout_indices = jnp.arange(Config.HORIZON)
        phys_indices = rollout_indices + Config.WARMUP_STEPS + 5
        scan_keys = jax.random.split(key, Config.HORIZON)
        
        # We also pass a running observation state as a carry: start with zeros for the perceptual feedback
        B = Config.BATCH_SIZE
        initial_weighted_belief = jnp.zeros((B, 4))
        initial_emd_intensities = jnp.zeros((B, Config.N_EMD_PIX))
        
        # Convert SLAM pose from physical 2m space → hornet physical metres for Instar routing
        # slam_pose[:2] = (x, y) in physical SLAM (= hornet + 1.0); slam_pose[2] = heading
        slam_xy_hornet = (slam_pose[:2] - _SLAM_OFFSET_TRAIN) / env._slam_scale
        slam_pose_hornet = jnp.concatenate([slam_xy_hornet, slam_pose[2:3]])  # (3,)
        frozen_pose  = jnp.broadcast_to(slam_pose_hornet, (B, 3))    # same pose for all agents
        frozen_surp  = jnp.full((B,), slam_surprise)                  # same surprise for all agents
        
        scan_inputs = (rollout_indices, phys_indices, scan_keys)
        init_carry = (start_state, initial_weighted_belief, initial_emd_intensities)

        def scan_fn(carry, xs): 
            curr_full, curr_weighted_belief, prev_emd_intensities = carry
            r_idx, p_idx, step_key = xs
            
            curr_robot = curr_full[0]
            start_robot = start_state[0]
            
            # SLAM-estimated position and heading at step t
            disp_x = curr_robot[:, 0] - start_robot[:, 0]
            disp_z = curr_robot[:, 1] - start_robot[:, 1]
            disp_th = curr_robot[:, 2] - start_robot[:, 2]
            
            slam_x_t = slam_xy_hornet[0] + disp_x
            slam_z_t = slam_xy_hornet[1] + disp_z
            slam_th_t = slam_pose_hornet[2] + disp_th
            
            # Construct the SLAM-based observation robot state relative to the target
            obs_robot = curr_robot
            obs_robot = obs_robot.at[:, 0].set(slam_x_t - target_xy[:, 0])
            obs_robot = obs_robot.at[:, 1].set(slam_z_t - target_xy[:, 1])
            # Reconstruct absolute body pitch: SLAM heading + 1.0 rad
            body_pitch_est = slam_th_t + 1.0
            wrapped_theta = jnp.mod(body_pitch_est + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs_robot = obs_robot.at[:, 2].set(wrapped_theta)

            scaled_obs = symlog(obs_robot)
            
            # --- Noise Injection (Sensor Model) ---
            noise_sigma = Config.OBS_NOISE_SIGMA
            obs_noise = jax.random.normal(step_key, shape=scaled_obs.shape) * noise_sigma
            noisy_obs = scaled_obs + obs_noise
            
            # --- Concat dynamic state with the weighted perceptual belief ---
            combined_obs = jnp.concatenate([noisy_obs, curr_weighted_belief], axis=-1)  # (B, 12)
            
            # --- GENERATE ACTION NOISE ---
            key_noise, key_step = jax.random.split(step_key)
            action_noise = jax.random.normal(key_noise, shape=(B, 4)) * Config.ACTION_NOISE_SIGMA

            # Compute SLAM position and sensor heading for dorsal/ventral pathways
            slam_u_batch = curr_robot[:, 0] * env._slam_scale + 1.0
            slam_v_batch = curr_robot[:, 1] * env._slam_scale + 1.0
            slam_pos_batch = jnp.stack([slam_u_batch, slam_v_batch], axis=-1)
            sensor_heading_batch = curr_robot[:, 2] - 1.0

            # --- DORSAL STREAM: Hassenstein-Reichardt EMD Pipeline ---
            # 1. Render coarse ommatidial intensities (32 pixels, distance-based)
            curr_emd_intensities, _ = jax.vmap(compute_emd_intensities, in_axes=(0, 0, None))(
                slam_pos_batch, sensor_heading_batch, env._segments
            )
            # 2. Speed-dependent temporal frequency adaptation
            vx = curr_robot[:, 4]
            vz = curr_robot[:, 5]
            speed = jnp.sqrt(vx**2 + vz**2 + 1e-8)
            tau = Config.EMD_TAU_BASE / (1.0 + Config.EMD_TAU_SPEED_GAIN * speed)
            emd_dt = Config.DT * Config.SIM_SUBSTEPS
            alpha_adaptive = jnp.exp(-emd_dt / tau)
            
            # 3. Reichardt correlate + LPTC pool: centering (HS) + looming (LGMD)
            emd_signals = jax.vmap(compute_emd_signals, in_axes=(0, 0, 0))(
                prev_emd_intensities, curr_emd_intensities, alpha_adaptive
            )

            # 3. Policy Inference
            batched_network = jax.vmap(ac_model.apply, in_axes=(0, 0, 0, None, None, 0, None, None, 0))
            mods, u_brain, _ = batched_network(
                params, combined_obs, action_noise, SOG_v_mem, Config.K_REPEL, emd_signals, Config.K_FLOW, Config.K_LOOM, slam_pos_batch
            )
            
            # --- LYAPUNOV HOVER & ATTENTION GATING (DNAG) ---
            # Use real SLAM surprise (frozen for this 32-step horizon).
            # Floor with positional distance so DNAG engages even before SLAM warms up.
            slam_pos_t = jnp.stack([slam_x_t, slam_z_t], axis=-1)
            dist_floor = jnp.clip(
                jnp.sqrt(jnp.sum((slam_pos_t - target_xy)**2, axis=-1) + 1e-8) * 2.0,
                0.0, 1.0
            )
            sim_surprise = jnp.maximum(frozen_surp, dist_floor)

            # Hover modulations: dedicated hover specialist (trained ICNN + BiologicalKinematicMap).
            # hover_fixed_params is closed over and treated as a constant by JAX JIT —
            # it NEVER receives gradients from the SHAC loss.
            # noisy_obs is 8D (physical state only), matching the hover specialist's obs space.
            if _use_hover_specialist:
                hover_mods, _, _ = batched_hover_network(
                    hover_fixed_params, noisy_obs, jnp.zeros((B, 4)), SOG_v_mem, Config.K_REPEL, emd_signals, Config.K_FLOW, Config.K_LOOM, slam_pos_batch
                )
            else:
                # Fallback: velocity-zeroed policy (if hover_params.pkl was not found)
                hover_robot   = curr_robot.at[:, 4:8].set(0.0)
                hover_robot   = hover_robot.at[:, 0].set(slam_x_t - target_xy[:, 0])
                hover_robot   = hover_robot.at[:, 1].set(slam_z_t - target_xy[:, 1])
                hover_wrapped = jnp.mod(hover_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
                hover_obs_raw = hover_robot.at[:, 2].set(hover_wrapped)
                hover_obs = jnp.concatenate([symlog(hover_obs_raw), curr_weighted_belief], axis=-1)
                hover_mods, _, _ = batched_network(
                    params, hover_obs, jnp.zeros((B, 4)), SOG_v_mem, Config.K_REPEL, emd_signals, Config.K_FLOW, Config.K_LOOM, slam_pos_batch
                )

            # Fully differentiable attention gate blending.
            # DNAG_MIN_ALPHA enforces a hover-specialist floor: even when SLAM
            # surprise is near zero (familiar territory / near target), the hover
            # specialist always contributes at least MIN_ALPHA of the action.
            # This prevents the SHAC policy's random critic from fully overriding
            # the hover-stable ICNN weights during early training.
            blended_mods, alpha = jax.vmap(differentiable_attention_gate)(sim_surprise, mods, hover_mods)
            alpha_floored = jnp.maximum(alpha, Config.DNAG_MIN_ALPHA)
            blended_mods  = (1.0 - alpha_floored) * mods + alpha_floored * hover_mods
            
            # 4. Environment Step (Physics uses the blended passivity-preserving actions)
            next_full, f_actual, _, _, _ = env.step_batch(curr_full, blended_mods, step_idx=p_idx)
            
            # --- PERCEPTUAL STREAM INGESTION (460 Hz Instar routing) ---
            # Use real SLAM pose + small noise as the pose_belief fed into Instar.
            key_stdp, _ = jax.random.split(key_step)
            slam_pose_t = jnp.stack([slam_x_t, slam_z_t, slam_th_t], axis=-1)
            pose_belief = slam_pose_t + jax.random.normal(key_step, shape=(B, 3)) * 0.01

            norm_csnn = jnp.broadcast_to(vis_csnn, (B, 256))
            norm_stdp  = jnp.broadcast_to(vis_stdp, (B, 256))
            visual_features = (norm_csnn, norm_stdp)
            
            # Update Instar Weights & Project weighted belief for the next control step
            # CRITICAL: Normalize u_brain from raw Newtons to [-1,1] saturated commands
            # before feeding as the Instar Hebbian target signal.  Raw forces (e.g. 50 N)
            # cause dW = eta*y*x ≈ 5*x per step → weight explosion → NaN in 32 steps.
            u_brain_saturated = jnp.tanh(u_brain / Config.FORCE_NORMALIZER)
            next_full, next_weighted_belief = env.ingest_perceptual_streams(
                next_full, pose_belief, visual_features, u_brain_saturated
            )
            
            # 5. Reward Calculation (Dynamic Waypoints)
            target_state_batch = jnp.zeros((B, 8))
            target_state_batch = target_state_batch.at[:, :2].set(target_xy)
            target_state_batch = target_state_batch.at[:, 2].set(Config.TARGET_STATE[2])
            target_state_batch = target_state_batch.at[:, 3].set(Config.TARGET_STATE[3])
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain, pbt_weights, target=target_state_batch)

            # --- OBSTACLE COLLISION PENALTY (differentiable gradient signal) ---
            # Convert physical (x, z) → SLAM space and compute signed distance to obstacles.
            # Positive = outside all obstacles, Negative = penetrating (inside).
            _slam_sc  = env._slam_scale
            _slam_off = 1.0
            cur_su = curr_robot[:, 0] * _slam_sc + _slam_off  # (B,) SLAM x
            cur_sz = curr_robot[:, 1] * _slam_sc + _slam_off  # (B,) SLAM z

            def _signed_dist(sx, sz):
                """Minimum signed distance from point to any obstacle rectangle.
                Positive = outside all obstacles; Negative = penetrating."""
                def _one_obs(obs):
                    x0, y0, x1, y1 = obs[0], obs[1], obs[2], obs[3]
                    is_inside = (sx >= x0) & (sx <= x1) & (sz >= y0) & (sz <= y1)
                    # Penetration depth (negative signed distance when inside)
                    depth = -jnp.minimum(
                        jnp.minimum(sx - x0, x1 - sx),
                        jnp.minimum(sz - y0, y1 - sz)
                    )
                    # Euclidean distance to nearest rectangle point when outside
                    cx = jnp.clip(sx, x0, x1)
                    cy = jnp.clip(sz, y0, y1)
                    outside = jnp.sqrt((sx - cx)**2 + (sz - cy)**2 + 1e-8)
                    return jnp.where(is_inside, depth, outside)
                dists = jax.vmap(_one_obs)(obstacles)  # (N_OBS,)
                return jnp.min(dists)

            min_obs_dists = jax.vmap(_signed_dist)(cur_su, cur_sz)  # (B,)
            # Penalty: activated inside safety margin (0.5 SLAM units ≈5 cm physical)
            obs_violation = jax.nn.relu(Config.OBS_SAFETY_SLAM - min_obs_dists)  # (B,)
            obs_penalty   = Config.OBS_PENALTY_WEIGHT * obs_violation            # (B,)

            # --- SCALE Force ---
            raw_diff = u_brain[:, :3] - f_actual[:, :3]
            norm_diff = raw_diff / Config.FORCE_NORMALIZER[:3]
            force_err_norm = jnp.mean(jnp.square(symlog(norm_diff)))
            
            # Apply Loss Weight
            loss_t = -rew_scaled + obs_penalty + (Config.AUX_LOSS_WEIGHT * force_err_norm)
            force_err_raw = jnp.mean(raw_diff**2)
            
            step_metrics = (
                loss_t, rew_scaled, force_err_raw, 
                met['rew'], met['pos'], 
                met['ang_th'], met['ang_ab'], 
                met['vel_lin'], met['vel_ang']
            )
            return (next_full, next_weighted_belief, curr_emd_intensities), step_metrics

        (final_full, final_weighted_belief, _), step_results = jax.lax.scan(
            scan_fn, init_carry, scan_inputs
        )
        
        (losses, rewards_scaled, f_errs, real_rews, m_pos, 
         m_ang_th, m_ang_ab, m_vel_lin, m_vel_ang) = step_results

        warmup_mask = rollout_indices >= Config.WARMUP_STEPS
        losses = jnp.where(warmup_mask[:, None], losses, 0.0)

        discounts = Config.GAMMA ** jnp.arange(Config.HORIZON)
        weighted_loss = jnp.dot(discounts, losses)
        
        final_robot = final_full[0]
        final_rel = final_robot.at[:, 0].set(final_robot[:, 0] - target_xy[:, 0])
        final_rel = final_rel.at[:, 1].set(final_robot[:, 1] - target_xy[:, 1])
        f_wrapped_th = jnp.mod(final_rel[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        final_obs = symlog(final_rel.at[:, 2].set(f_wrapped_th))
        
        # Concat final observation for the critic bootstrap
        final_combined_obs = jnp.concatenate([final_obs, final_weighted_belief], axis=-1)

        _, _, final_val_actor = jax.vmap(ac_model.apply)(params, final_combined_obs)
        final_val_actor = jnp.squeeze(final_val_actor)
        
        actor_term = jnp.mean(weighted_loss - (Config.GAMMA**Config.HORIZON * final_val_actor))

        final_val_target = jax.lax.stop_gradient(final_val_actor)
        discounted_return = jnp.dot(discounts, rewards_scaled) + (Config.GAMMA**Config.HORIZON * final_val_target)
        
        start_robot = start_state[0]
        start_rel = start_robot.at[:, 0].set(start_robot[:, 0] - target_xy[:, 0])
        start_rel = start_rel.at[:, 1].set(start_robot[:, 1] - target_xy[:, 1])
        s_wrapped_th = jnp.mod(start_rel[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        start_obs = symlog(start_rel.at[:, 2].set(s_wrapped_th))
        start_combined_obs = jnp.concatenate([start_obs, initial_weighted_belief], axis=-1)

        _, _, start_val = jax.vmap(ac_model.apply)(params, start_combined_obs)
        start_val = jnp.squeeze(start_val)
        
        critic_loss = optax.huber_loss(start_val, discounted_return, delta=1.0)
        critic_loss = jnp.mean(critic_loss)
        
        total_loss = actor_term + (0.5 * critic_loss)
        
        logs = {
            'rew': jnp.mean(real_rews),
            'rew_per_agent': jnp.mean(real_rews, axis=0),
            'ferr': jnp.mean(f_errs),
            'pos': jnp.mean(m_pos),
            'pos_per_agent': jnp.mean(m_pos, axis=0), 
            'ang_th': jnp.mean(m_ang_th),
            'ang_ab': jnp.mean(m_ang_ab),
            'vel_lin': jnp.mean(m_vel_lin),
            'vel_ang': jnp.mean(m_vel_ang),
            'act_loss': actor_term,
            'crit_loss': critic_loss
        }
        return total_loss, (logs, final_full)

    # Constant needed inside loss_fn (mirrors env._SLAM_OFFSET)
    _SLAM_OFFSET_TRAIN = 1.0  # physical: hornet ±ARENA_W → SLAM [0, 2m]

    @jax.jit
    def update(params, opt_state, full_state, pbt_state, key, slam_pose, slam_surprise, obstacles, vis_csnn, vis_stdp, SOG_v_mem, target_xy):
        key_loss, key_next = jax.random.split(key)
        
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, full_state, pbt_state.weights, key_loss, slam_pose, slam_surprise, obstacles, vis_csnn, vis_stdp, SOG_v_mem, target_xy
        )
        
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        current_score = -1.0 * (logs['pos_per_agent'] * 100.0)

        new_running = 0.8 * pbt_state.running_reward + 0.2 * current_score
        new_pbt_state = pbt_state._replace(running_reward=new_running)

        return new_params, new_opt, loss, logs, next_state, new_pbt_state, key_next

    # Initialize SOG for training tracking
    sog_system = SpikingOccupancyGrid(map_size_m=2.0, res=0.04, offset_m=0.0, v_max=1.0)
    sog_state = sog_system.init_state()

    print(f"=== Starting Training: Step {start_step} to {Config.TOTAL_UPDATES} ===")
    rng, key_reset = jax.random.split(rng)
    curr_state = env.reset(key_reset, Config.BATCH_SIZE)
    rng, key_tgt = jax.random.split(rng)
    target_xy = jax.random.uniform(key_tgt, (Config.BATCH_SIZE, 2), minval=-0.8, maxval=0.8)
    
    # --- Initialise SLAM System ---
    # One SNNSLAMSystem tracks agent-0's trajectory and provides real pose + surprise
    # to the DNAG gate.  It runs OUTSIDE the JAX JIT loop (Python-native).
    print("---> Initialising SNNSLAMSystem (agent-0 tracker)...")
    slam_system = SNNSLAMSystem(jax.random.PRNGKey(7), n_depth=N_DEPTH)
    slam_system.reset(1)

    # Bootstrap SLAM from agent-0's initial pose (centre of 10m room)
    init_robot0  = np.array(curr_state[0][0])            # (8,) hornet state
    init_slam_x  = float(init_robot0[0]) * env._slam_scale + 1.0
    init_slam_z  = float(init_robot0[1]) * env._slam_scale + 1.0
    init_slam_th = float(init_robot0[2]) - 1.0  # sensor heading (facing forward)
    slam_system.initialize_pose(
        jnp.array([[init_slam_x, init_slam_z]]),
        jnp.array([init_slam_th]),
    )

    # SLAM state carried between outer loop iterations
    slam_pose     = jnp.array([init_slam_x, init_slam_z, init_slam_th])  # (3,)
    slam_surprise = 0.0                                                    # float
    slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)                  # event delta baseline
    slam_vis_csnn = jnp.zeros(256)
    slam_vis_stdp = jnp.zeros(256)
    print(f"    -> SLAM initialised at ({init_slam_x:.2f}, {init_slam_z:.2f}) m, sensor heading {init_slam_th:.2f} rad")
    
    # Helper: regenerate arena + reset SLAM at episode boundaries
    def _reset_slam_for_new_arena(key, curr_agent0_state):
        """
        Called at forced-reset boundaries (PBT + periodic RESET_INTERVAL).
        1. Regenerates obstacle room (new episode = new room).
        2. Resets SNNSLAMSystem and re-initialises from agent-0's current pose.
        3. Clears the event-camera intensity baseline.
        Returns updated slam_pose, slam_surprise, slam_prev_int.
        """
        nonlocal slam_prev_int, slam_vis_csnn, slam_vis_stdp, sog_state
        if key is not None:
            env.regenerate_arena(key=key)
        env._prev_robot_state = None

        # Re-initialise SLAM from agent-0's current pose in the new room
        r0 = np.array(curr_agent0_state)
        new_slam_x  = float(r0[0]) * env._slam_scale + 1.0
        new_slam_z  = float(r0[1]) * env._slam_scale + 1.0
        new_slam_th = float(r0[2]) - 1.0  # sensor heading (facing forward)

        slam_system.reset(1)
        slam_system.initialize_pose(
            jnp.array([[new_slam_x, new_slam_z]]),
            jnp.array([new_slam_th]),
        )
        slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)  # clear event baseline
        slam_vis_csnn = jnp.zeros(256)
        slam_vis_stdp = jnp.zeros(256)
        sog_state = sog_system.init_state()

        new_slam_pose = jnp.array([new_slam_x, new_slam_z, new_slam_th])
        print(f"    ---> New arena + SLAM reset: pose ({new_slam_x:.2f}, {new_slam_z:.2f}) m, sensor heading {new_slam_th:.2f} rad")
        return new_slam_pose, 0.0  # fresh room → surprise starts at 0

    def _reinit_slam_on_crash(curr_agent0_state):
        """
        Called when agent-0 crashes and resets individually.
        Resets and re-initialises the SLAM tracker to match the fresh spawn position,
        without regenerating the obstacle distribution of the arena.
        """
        nonlocal slam_prev_int, slam_vis_csnn, slam_vis_stdp, sog_state
        env._prev_robot_state = None
        r0 = np.array(curr_agent0_state)
        new_slam_x  = float(r0[0]) * env._slam_scale + 1.0
        new_slam_z  = float(r0[1]) * env._slam_scale + 1.0
        new_slam_th = float(r0[2]) - 1.0

        slam_system.reset_pose_only(1)
        slam_system.initialize_pose(
            jnp.array([[new_slam_x, new_slam_z]]),
            jnp.array([new_slam_th]),
        )
        slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)
        slam_vis_csnn = jnp.zeros(256)
        slam_vis_stdp = jnp.zeros(256)
        sog_state = sog_system.init_state()

        new_slam_pose = jnp.array([new_slam_x, new_slam_z, new_slam_th])
        print(f"    [SLAM RESET ON CRASH] Re-initialized at ({new_slam_x:.2f}, {new_slam_z:.2f}) m")
        return new_slam_pose, 0.0

    # --- JIT Compilation ---
    print("-----> Compiling JAX Update...")
    t0 = time.time()
    key_compile = jax.random.PRNGKey(0)
    _dummy_sog = jnp.zeros((50, 50))
    _ = update(params, opt_state, curr_state, pbt_state, key_compile, slam_pose, jnp.array(slam_surprise), env._obstacles, slam_vis_csnn, slam_vis_stdp, _dummy_sog, target_xy)
    print(f"---> Compilation Finished in {time.time() - t0:.2f}s")

    # --- Main Loop ---
    key_explore = jax.random.PRNGKey(999) 
    t_start = time.time()
    
    for i in range(start_step, Config.TOTAL_UPDATES):
        t0 = time.time() 

        if i == start_step:
            print(f"DEBUG: Env Target Theta: {env.target[2]:.4f}")
            print(f"DEBUG: Spawn Theta: {curr_state[0][0, 2]:.4f}")

        # 1. Update Step — keep a snapshot so we can roll back on NaN
        _prev_params   = params
        _prev_opt_state = opt_state

        params, opt_state, loss, logs, next_state, pbt_state, key_explore = update(
            params, opt_state, curr_state, pbt_state, key_explore,
            slam_pose, jnp.array(slam_surprise), env._obstacles, slam_vis_csnn, slam_vis_stdp, jnp.array(sog_state.v_mem),
            target_xy
        )

        # 1b. SLAM update (outside JIT, runs on agent-0's terminal state)
        # Compute sensor bundle from the final state of this horizon
        robot0_np = np.array(next_state[0][0])  # (8,) agent-0 hornet state
        env._prev_robot_state = np.array(curr_state[0][0])
        ev_jax, kin_jax, tof_jax, acc_jax, slam_prev_int = env.compute_slam_sensors(
            robot0_np, slam_prev_int, dt=Config.HORIZON * Config.SIM_SUBSTEPS * Config.DT
        )
        # Run SLAM steps (closed-loop with full memory + loop closure detection)
        # 3 steps align the SLAM time-scale with the 32-step physical horizon (0.069s)
        dt_horizon = Config.HORIZON * Config.SIM_SUBSTEPS * Config.DT
        dt_per_cann_step = dt_horizon / 3.0  # ≈ 0.02304 s per CANN step
        try:
            for _ in range(3):
                pose_est, _, _, _, _, debug_gates = slam_system.forward_step(
                    ev_jax, kin_jax, tof_jax, acc_t=acc_jax,
                    inject_drift=False, autopilot_on=(slam_surprise < 0.60),
                    dt=dt_per_cann_step
                )
            slam_pose     = jnp.array([
                float(pose_est[0, 0]),   # x in 2m space
                float(pose_est[0, 1]),   # y in 2m space
                float(pose_est[0, 2]),   # heading
            ])
            slam_surprise = float(1.0 - float(debug_gates['Raw_Match'][0]))
            slam_vis_csnn = jnp.array(debug_gates['Debug_Input_CSNN'][0])
            slam_vis_stdp = jnp.array(debug_gates['Debug_Input_STDP'][0])
        except Exception as _slam_err:
            # Graceful fallback: keep previous SLAM values
            print(f"    [SLAM] non-fatal error at step {i}: {_slam_err}")
            
        # Update SOG for training agent-0 (outside JIT)
        try:
            s_u = robot0_np[0] * env._slam_scale + 1.0
            s_v = robot0_np[1] * env._slam_scale + 1.0
            s_th = robot0_np[2] - 1.0
            _hit, _free = _get_ray_indices(
                s_u, s_v, s_th,
                np.array(tof_jax[0]), [-np.pi/4, 0.0, np.pi/4],
                res=sog_system.res, grid_size=sog_system.grid_w, offset_m=sog_system.offset_m,
            )
            sog_state = sog_system.update(sog_state, jnp.array(_hit), jnp.array(_free))
        except Exception as _sog_err:
            print(f"    [SOG] non-fatal error at step {i}: {_sog_err}")
    
        # 2. Stability Checks
        if jnp.isnan(loss):
            print(f"!!! CRITICAL: NaN detected at step {i} !!! Rolling back params + resetting batch.")
            params    = _prev_params     # restore last-known-good weights
            opt_state = _prev_opt_state  # restore last-known-good optimizer state
            rng, k_res = jax.random.split(rng)
            curr_state = env.reset(k_res, Config.BATCH_SIZE)
            slam_pose, slam_surprise = _reinit_slam_on_crash(curr_state[0][0])
            continue
        
        r_state = next_state[0]
        
        # --- Environment Reset Logic ---
        is_nan = jnp.isnan(r_state).any(axis=1)
        is_oor = (jnp.abs(r_state[:, 0]) > Config.ARENA_W) | (jnp.abs(r_state[:, 1]) > Config.ARENA_W)
        # Obstacle collision: check in SLAM space using numpy (fast, outside JIT)
        _obs_np     = np.array(env._obstacles) if env._obstacles is not None else np.zeros((0, 4))
        _slam_xs_np = np.array(r_state[:, 0]) * float(env._slam_scale) + 1.0
        _slam_zs_np = np.array(r_state[:, 1]) * float(env._slam_scale) + 1.0
        def _in_any_obs(sx, sz):
            if len(_obs_np) == 0: return False
            return bool(np.any(
                (_obs_np[:, 0] <= sx) & (_obs_np[:, 2] >= sx) &
                (_obs_np[:, 1] <= sz) & (_obs_np[:, 3] >= sz)
            ))
        is_in_obs = jnp.array([_in_any_obs(sx, sz)
                                for sx, sz in zip(_slam_xs_np, _slam_zs_np)])
        is_crashed = is_oor | is_in_obs
        reset_mask = is_nan | is_crashed
    
        rng, k_res = jax.random.split(rng)
        fresh_state = env.reset(k_res, Config.BATCH_SIZE)
    
        curr_state = jax.tree.map(
            lambda x, y: jnp.where(jnp.reshape(reset_mask, (-1,) + (1,)*(x.ndim-1)), y, x),
            next_state, fresh_state
        )

        # Randomize target for crashed agents
        rng, k_fresh_tgt = jax.random.split(rng)
        fresh_target_xy = jax.random.uniform(k_fresh_tgt, (Config.BATCH_SIZE, 2), minval=-0.8, maxval=0.8)
        target_xy = jnp.where(reset_mask[:, None], fresh_target_xy, target_xy)

        # 🌟 CRITICAL FIX: If agent 0 (tracked by SLAM) crashed and reset,
        # we must reset the SLAM tracker to match its new spawn position!
        if reset_mask[0]:
            slam_pose, slam_surprise = _reinit_slam_on_crash(curr_state[0][0])

        # 3. Telemetry & Logging
        dt_epoch = time.time() - t0              
        total_elapsed = time.time() - t_start    

        sample_robot = next_state[0][0] 
        sample_osc = jax.tree.map(lambda x: x[0], next_state[2])

        wrapped_theta = jnp.mod(sample_robot[2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        obs_sample = sample_robot.at[2].set(wrapped_theta)
        scaled_sample = symlog(obs_sample)
        
        # Pad observations for sample telemetry CPG modulations
        dummy_weighted = jnp.zeros(4)
        combined_sample = jnp.concatenate([scaled_sample, dummy_weighted], axis=-1)

        params_sample = jax.tree.map(lambda x: x[0], params)
        sample_mods, _, _ = ac_model.apply(params_sample, combined_sample)

        angles, _, _, _ = get_wing_kinematics(sample_osc, unpack_action(sample_mods))
        str_angle, dev_angle, pit_angle = angles
    
        mean_x = jnp.mean(curr_state[0][:, 0])
        mean_z = jnp.mean(curr_state[0][:, 1])
    
        lin_vels = next_state[0][:, 4:6]
        lin_mag = jnp.mean(jnp.sqrt(jnp.sum(lin_vels**2, axis=1)))
        ang_vels = next_state[0][:, 6:8]
        
        raw_hum_energy = jnp.mean(jnp.sum(ang_vels**2, axis=1))
        thorax_vel = next_state[0][:, 6]
        thorax_mag = jnp.mean(jnp.abs(thorax_vel))

        # --- A. PBT EVOLUTION ---
        performed_pbt = False
        if i % Config.PBT_INTERVAL == 0 and i > 0:
            print(f"---> PBT EVOLUTION (Step {i})")
            rng, k_pbt = jax.random.split(rng)

            best_idx = jnp.argmax(pbt_state.running_reward)
            best_score = pbt_state.running_reward[best_idx]
            print(f"    Best Score: {best_score:.2f} cm | Weights: {pbt_state.weights[best_idx]}")
        
            params, opt_state, pbt_state = pbt_evolve(
                k_pbt, 
                params, 
                opt_state, 
                pbt_state,
                perturb_factor=Config.PBT_PERTURB_FACTOR,
                truncate_fraction=Config.PBT_TRUNCATE_FRACTION
            )
            performed_pbt = True 

        # --- B. FORCED RESET (with arena + SLAM regeneration) ---
        if (i % Config.RESET_INTERVAL == 0 and i > 0) or performed_pbt:
            if performed_pbt:
                print("    -> PBT Mutation applied. Forcing Environment + Arena Reset.")
            rng, k_arena = jax.random.split(rng)
            env.regenerate_arena(key=k_arena)
            curr_state = env.reset(rng, Config.BATCH_SIZE)
            slam_pose, slam_surprise = _reset_slam_for_new_arena(
                None, curr_state[0][0]  # agent-0's new spawn state
            )
            rng, k_tgt = jax.random.split(rng)
            target_xy = jax.random.uniform(k_tgt, (Config.BATCH_SIZE, 2), minval=-0.8, maxval=0.8)

        print(f"⚡ [SHAC EPOCH {i:04d}] Loss: {loss:.2e} (Act: {logs['act_loss']:.1e}, Crit: {logs['crit_loss']:.1e}) | Reward: {logs['rew']:.2f} | Time: {dt_epoch:.1f}s (Total: {total_elapsed/60:.1f}m)\n"
              f"   ├── Target Tracking Errors: Pos: {logs['pos']:.2f}m | Pitch: {logs['ang_th']:.2f} rad | Abdomen: {logs['ang_ab']:.2f} rad | MeanPos: [{mean_x:+.2f}, {mean_z:+.2f}]m\n"
              f"   └── Biomechanics & Forces : LinVel: {logs['vel_lin']:.2f} m/s | AngVel: {logs['vel_ang']:.2f} rad/s | WingForce Err: {logs['ferr']:.3f} mN\n"
              f"   └── Energetics & Flight   : Muscle Energy: {raw_hum_energy:.0f} J | Thorax AngVel: {thorax_mag:.2f} rad/s")
    
        if i % Config.VIS_INTERVAL == 0 or i == start_step:
            ckpt_path = os.path.join(Config.CKPT_DIR, f"shac_params_{i}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    'params': params, 
                    'opt_state': opt_state,
                    'pbt_state': pbt_state 
                }, f)
            print(f"--> Saved Checkpoint: {ckpt_path}")
            run_visualization(
                env, params, i, vis_step_fn,
                pbt_state=pbt_state,
                curr_state=curr_state,
                slam_system=slam_system,
                sog_state=sog_state,
                target_xy=target_xy,
                slam_pose=slam_pose,
                slam_surprise=slam_surprise
            )

# Global Visualization Step
@partial(jax.jit, static_argnums=(0,))
def vis_step_fn(env, curr_state, curr_params, step_idx, slam_surprise, hover_params_single, slam_pose, vis_csnn, vis_stdp, SOG_v_mem, K_repel, K_flow, target_xy, prev_emd_intensities):
    r_st = curr_state[0]
    B = r_st.shape[0]
    
    # Convert SLAM pose from physical SLAM space → hornet physical metres for Instar routing
    slam_xy_hornet = (slam_pose[:, :2] - 1.0) / env._slam_scale
    pose_belief = jnp.concatenate([slam_xy_hornet, slam_pose[:, 2:3]], axis=-1)
    
    # 1. Prepare Observation using SLAM-estimated position and heading relative to the target
    obs_v = r_st
    obs_v = obs_v.at[:, 0].set(pose_belief[:, 0] - target_xy[:, 0])
    obs_v = obs_v.at[:, 1].set(pose_belief[:, 1] - target_xy[:, 1])
    # Reconstruct absolute body pitch: SLAM heading + 1.0 rad
    body_pitch_est = pose_belief[:, 2] + 1.0
    wrapped_th = jnp.mod(body_pitch_est + jnp.pi, 2 * jnp.pi) - jnp.pi
    obs_v = obs_v.at[:, 2].set(wrapped_th)
    scaled_obs = symlog(obs_v)
    
    # 2. Ingest visual-spatial belief features
    norm_csnn = vis_csnn
    norm_stdp = vis_stdp
    visual_features = (norm_csnn, norm_stdp)
    
    next_state, weighted_belief = env.ingest_perceptual_streams(
        curr_state, pose_belief, visual_features, jnp.zeros((B, 4))
    )
    
    combined_obs = jnp.concatenate([scaled_obs, weighted_belief], axis=-1)
    
    # Compute SLAM position and sensor heading for dorsal/ventral pathways
    slam_u_batch = r_st[:, 0] * env._slam_scale + 1.0
    slam_v_batch = r_st[:, 1] * env._slam_scale + 1.0
    slam_pos_batch = jnp.stack([slam_u_batch, slam_v_batch], axis=-1)
    sensor_heading_batch = r_st[:, 2] - 1.0

    # --- DORSAL STREAM: Hassenstein-Reichardt EMD Pipeline ---
    curr_emd_intensities, _ = jax.vmap(compute_emd_intensities, in_axes=(0, 0, None))(
        slam_pos_batch, sensor_heading_batch, env._segments
    )
    # Speed-dependent temporal frequency adaptation
    vx = r_st[:, 4]
    vz = r_st[:, 5]
    speed = jnp.sqrt(vx**2 + vz**2 + 1e-8)
    tau = Config.EMD_TAU_BASE / (1.0 + Config.EMD_TAU_SPEED_GAIN * speed)
    emd_dt = Config.DT * Config.SIM_SUBSTEPS
    alpha_adaptive = jnp.exp(-emd_dt / tau)
    
    emd_signals = jax.vmap(compute_emd_signals, in_axes=(0, 0, 0))(
        prev_emd_intensities, curr_emd_intensities, alpha_adaptive
    )

    # 3. Policy Inference
    mods, _, _ = ac_model.apply(
        curr_params, combined_obs, None, SOG_v_mem, K_repel, emd_signals, K_flow, Config.K_LOOM, slam_pos_batch
    )
    
    # --- LYAPUNOV HOVER & ATTENTION GATING (DNAG) ---
    dist_floor = jnp.clip(
        jnp.sqrt(jnp.sum((r_st[:, :2] - target_xy)**2, axis=-1) + 1e-8) * 2.0,
        0.0, 1.0
    )
    sim_surprise = jnp.maximum(slam_surprise, dist_floor)
    
    if hover_params_single is not None:
        hover_mods, _, _ = hover_ac_model.apply(
            hover_params_single, scaled_obs, None, SOG_v_mem, K_repel, emd_signals, K_flow, Config.K_LOOM, slam_pos_batch
        )
    else:
        # Fallback: velocity-zeroed policy
        hover_robot = r_st.at[:, 4:8].set(0.0)
        hover_wrapped = jnp.mod(hover_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        hover_obs_raw = hover_robot.at[:, 2].set(hover_wrapped)
        hover_obs = jnp.concatenate([symlog(hover_obs_raw), weighted_belief], axis=-1)
        hover_mods, _, _ = ac_model.apply(
            curr_params, hover_obs, None, SOG_v_mem, K_repel, emd_signals, K_flow, Config.K_LOOM, slam_pos_batch
        )
        
    blended_mods, alpha = jax.vmap(differentiable_attention_gate)(sim_surprise, mods, hover_mods)
    alpha_floored = jnp.maximum(alpha, Config.DNAG_MIN_ALPHA)
    blended_mods = (1.0 - alpha_floored) * mods + alpha_floored * hover_mods
    
    # 4. Env Step
    next_state, _, f_nodal, w_pose, h_marker = env.step_batch(next_state, blended_mods, step_idx=step_idx)
    
    # Compute SOG repulsive force
    f_repel = compute_sog_repulsive_force(slam_pos_batch, SOG_v_mem)
    f_repel_scaled = K_repel * f_repel
    
    # Compute EMD-derived centering signal for telemetry
    centering_signal = emd_signals[:, 0]
    flow_corr = K_flow * centering_signal
    
    return next_state, f_nodal, w_pose, h_marker, alpha_floored, f_repel_scaled, flow_corr, curr_emd_intensities

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="checkpoints_shac", 
                        help="Directory to save checkpoints and visuals")
    parser.add_argument("--gpu", action="store_true", 
                        help="Enable GPU (Removes CPU forcing)")
    parser.add_argument("--reset_PBTweights", action="store_true", 
                        help="force-overwrite PBT weights from Config (One-time fix)")
    args = parser.parse_args()

    abs_dir = os.path.abspath(args.dir)
    Config.CKPT_DIR = abs_dir
    Config.VIS_DIR = abs_dir
    
    print(f"--> OUTPUT DIRECTORY: {Config.CKPT_DIR}")

    if args.gpu:
        if "JAX_PLATFORMS" in os.environ:
            del os.environ["JAX_PLATFORMS"]
        print("--> MODE: GPU Enabled (JAX Default)")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        print("--> MODE: Force CPU (Use --gpu to enable GPU)")

    train()