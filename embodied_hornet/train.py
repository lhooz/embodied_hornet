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
from .neural_idapbc import policy_network_icnn, differentiable_attention_gate, unpack_action, ScaleConfig
from .env import FlyEnv

# --- BASE MODULES FROM hornetRL SIBLING REPO ---
from hornetRL.fluid_surrogate import JaxSurrogateEngine
from hornetRL.fly_system import FlappingFlySystem, PhysParams
from hornetRL.neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from hornetRL.pbt_manager import init_pbt_state, pbt_evolve

# --- SLAM SYSTEM FROM neuro-symbolic-slam SUBMODULE ---
# (sys.path for src/ is configured in embodied_hornet/__init__.py)
from snn_slam_system import SNNSLAMSystem, N_DEPTH
from sparse_forest import N_PIXELS, compute_tof_distance

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
    # 1m × 1m physical arena (SLAM coordinate: ±0.5m → [0.5, 9.5]m in 10m SLAM space)
    ARENA_W = 0.5

    # --- Time Scales ---
    DT = 3e-5               # Physics integration timestep (s)
    SIM_SUBSTEPS = 72       # Physics steps per Control Step
                            
    HORIZON = 32            # Trajectory horizon for Backpropagation (= 8 wingbeats @ 4 steps/beat)
    RESET_INTERVAL = 50     
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

    CURRICULUM_RATIO = 0.5
    
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
    VIS_INTERVAL = 200

    # --- Obstacle Collision ---
    OBS_PENALTY_WEIGHT = 50.0   # reward penalty per SLAM unit of penetration
    OBS_SAFETY_SLAM    = 0.5    # safety buffer in SLAM units (≈5 cm physical)

    WARMUP_STEPS = 1        

    FORCE_NORMALIZER = ScaleConfig.CONTROL_SCALE

# --- Observation Scaling SYMLOG ---
def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
def actor_critic_fn(combined_state, action_noise=None):
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
        action_noise=action_noise
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
def hover_actor_fn(physical_state, action_noise=None):
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
        action_noise=action_noise
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
def run_visualization(env, params, update_idx, vis_step_fn):
    print(f"--> Generating Visualization for Step {update_idx}...")

    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 8

    sim_data = {
        'states': [], 'wing_pose': [], 'nodal_forces': [],
        'le_marker': [], 'hinge_marker': [], 't': [],
        'slam_pos': [],   # true physical position in SLAM space
        'tof': [],        # (3,) ToF distances per frame (SLAM metres)
        'heading': [],    # theta (rad) per frame
    }
    
    rng = jax.random.PRNGKey(update_idx)
    state = env.reset(rng, 1) 
    
    active_props_batch = state[3] 
    real_props = jax.tree.map(lambda x: x[0], active_props_batch)
    params_single = jax.tree.map(lambda x: x[0], params)

    current_step_counter = 0

    for i in range(total_visual_frames):
        for _ in range(steps_per_frame):
            r_st = state[0]
            r_cpu = np.array(r_st[0])
            if np.isnan(r_cpu).any():
                print(f"!!! Visualization stopped early due to NaN !!!")
                break
            
            state, f_nodal, w_pose, h_marker = vis_step_fn(env, state, params_single, current_step_counter)
            current_step_counter += 1

        r_state_np = np.array(state[0][0])  # (8,) physical state of agent 0
        sim_data['states'].append(r_state_np)
        sim_data['t'].append(current_step_counter * Config.DT)
        # Convert physical (x, z) → SLAM (u, v) for nav panel
        _slam_scale  = env._slam_scale
        _slam_offset = 5.0
        slam_u = r_state_np[0] * _slam_scale + _slam_offset
        slam_v = r_state_np[1] * _slam_scale + _slam_offset
        sim_data['slam_pos'].append((slam_u, slam_v))
        # ToF: 3 beam distances from current SLAM position + heading
        _tof_jax = compute_tof_distance(
            jnp.array([slam_u, slam_v]), float(r_state_np[2]), env._segments
        )
        sim_data['tof'].append(np.array(_tof_jax))
        sim_data['heading'].append(float(r_state_np[2]))

        f_st = state[1]
        sim_data['le_marker'].append(np.array(f_st.marker_le[0]))
        sim_data['wing_pose'].append(np.array(w_pose[0]))
        sim_data['nodal_forces'].append(np.array(f_nodal[0]))
        sim_data['hinge_marker'].append(np.array(h_marker[0]))

    # -----------------------------------------------------------------------
    # Matplotlib Animation — two-panel layout
    # Left:  10m×10m navigation room (SLAM space) with obstacles + trajectory
    # Right: close-up wing mechanics (as before)
    # -----------------------------------------------------------------------
    fig, (ax_nav, ax_wing) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#0d0d0d')
    for ax in (ax_nav, ax_wing):
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # --- Left panel: navigation room ---
    ax_nav.set_xlim(0, 10)
    ax_nav.set_ylim(0, 10)
    ax_nav.set_aspect('equal')
    ax_nav.set_title('Navigation Room (SLAM Space)', color='white', fontsize=10)
    ax_nav.set_xlabel('X (m)', color='#aaaaaa')
    ax_nav.set_ylabel('Y (m)', color='#aaaaaa')

    # Draw room boundary
    room_rect = plt.Rectangle((0, 0), 10, 10, linewidth=2, edgecolor='#00ffcc', facecolor='none')
    ax_nav.add_patch(room_rect)

    # --- Draw obstacles as dynamic patches (re-colored on collision) ---
    obstacles_np = np.array(env._obstacles) if env._obstacles is not None else np.zeros((0, 4))
    obs_patch_list = []  # list of (obs_bbox, patch) for collision highlighting
    for obs in obstacles_np:
        x0, y0, x1, y1 = obs
        p = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, zorder=4)
        ax_nav.add_patch(p)
        obs_patch_list.append((obs, p))

    # Draw target in SLAM coords
    target_phys = np.array(Config.TARGET_STATE)
    _slam_scale  = env._slam_scale
    _slam_offset = 5.0
    tgt_u = target_phys[0] * _slam_scale + _slam_offset
    tgt_v = target_phys[1] * _slam_scale + _slam_offset
    ax_nav.plot(tgt_u, tgt_v, marker='*', markersize=14, color='#ffdd00',
                markeredgecolor='#ff8800', zorder=20, label='Target')

    # --- Trajectory + hornet + heading arrow (animated) ---
    traj_line, = ax_nav.plot([], [], '-', color='#00ff88', linewidth=1.0, alpha=0.6, zorder=5)
    hornet_dot, = ax_nav.plot([], [], 'o', color='#ff4444', markersize=7, zorder=15)
    heading_arr = ax_nav.quiver([], [], [], [], color='#ff8888', scale=20, width=0.004, zorder=16)

    # --- 3 ToF beam artists: [left, center, right] ---
    _beam_colours = ['#00aaff', '#ffff44', '#00aaff']
    tof_beam_artists = []
    for _bc in _beam_colours:
        _bl, = ax_nav.plot([], [], '-',  color=_bc, linewidth=1.5, alpha=0.85, zorder=12)
        _bm, = ax_nav.plot([], [], 'D',  color=_bc, markersize=4,  alpha=0.95, zorder=13)
        tof_beam_artists.append((_bl, _bm))

    # --- Camera FOV boundary dashes ---
    fov_left_line,  = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
    fov_right_line, = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)

    nav_time = ax_nav.text(0.02, 0.96, '', transform=ax_nav.transAxes,
                           color='#cccccc', fontsize=8, va='top', family='monospace')
    ax_nav.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)

    # --- Right panel: close-up wing mechanics (unchanged logic) ---
    ax_wing.set_aspect('equal')
    ax_wing.set_title('Wing Mechanics (Close-up)', color='white', fontsize=10)
    ax_wing.set_xlabel('X (m)', color='#aaaaaa')
    ax_wing.set_ylabel('Z (m)', color='#aaaaaa')
    ax_wing.grid(True, linestyle=':', alpha=0.2, color='#444444')

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
            heading_arr.set_offsets([[xs[-1], ys[-1]]])
            heading_arr.set_UVC([np.cos(r_th)], [np.sin(r_th)])

            # === SLAM sensor layer ===
            if frame < len(sim_data['tof']):
                tof_d = sim_data['tof'][frame]      # (3,) SLAM distances
                hdg   = sim_data['heading'][frame]  # theta (rad)
                cu, cv = xs[-1], ys[-1]

                # 3 ToF beams: left/center/right at ±45° from heading
                for bi, ((bl, bm), offset) in enumerate(
                        zip(tof_beam_artists, [-np.pi/4, 0.0, np.pi/4])):
                    ang = hdg + offset
                    hu  = cu + tof_d[bi] * np.cos(ang)
                    hv  = cv + tof_d[bi] * np.sin(ang)
                    bl.set_data([cu, hu], [cv, hv])
                    bm.set_data([hu], [hv])

                # FOV boundary (90° camera cone)
                fov_r = 6.0  # display range (SLAM m)
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

                # SLAM text overlay
                nav_time.set_text(
                    f'θ={hdg:.2f}r | ToF [{tof_d[0]:.1f}\u2502{tof_d[1]:.1f}\u2502{tof_d[2]:.1f}]m'
                )
            else:
                nav_time.set_text(f'T={t:.3f}s')

        # -- right panel: wing mechanics --
        ax_wing.set_xlim(rx - 0.06, rx + 0.06)
        ax_wing.set_ylim(rz - 0.06, rz + 0.06)

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
        patch_le.set_center((le_pos[0], le_pos[1]))
        patch_hinge.set_center((hinge_p[0], hinge_p[1]))
        pts = np.stack([wing_x, wing_z], axis=1)
        quiver.set_offsets(pts)
        quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
        time_text.set_text(f'T={t:.4f}s | Z={rz:.3f}m')

        return patch_thorax, patch_le, patch_hinge, traj_line, hornet_dot

    plt.tight_layout()
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
    batched_hover_network = jax.vmap(hover_ac_model.apply)

    # Validate hover specialist with a dry-run forward pass before JIT
    if _use_hover_specialist:
        try:
            _dummy_8d   = jnp.zeros((_hover_batch, 8))
            _dummy_act  = jnp.zeros((_hover_batch, 4))
            _test_mods, _, _ = batched_hover_network(hover_fixed_params, _dummy_8d, _dummy_act)
            print(f"    -> Hover specialist validated: mods shape {_test_mods.shape}")
        except Exception as _e:
            print(f"    -> WARNING: hover specialist forward-pass failed ({_e})")
            print(f"       Param keys: {list(hover_fixed_params.keys())}")
            print(f"       Falling back to velocity-zeroed policy.")
            _use_hover_specialist = False
            hover_fixed_params    = None

    def loss_fn(params, start_state, pbt_weights, key, slam_pose, slam_surprise, obstacles):
        """
        Computes the total loss over the trajectory horizon.
        Includes policy gradient, value function loss, and auxiliary force matching loss.

        slam_pose:     (3,) JAX array  [slam_x, slam_y, slam_heading] in 10m SLAM space.
                       Converted to hornet metres before being fed to the Instar routing.
        slam_surprise: scalar float  in [0, 1]  — 1.0 = fully novel scene, 0.0 = familiar.
        """
        rollout_indices = jnp.arange(Config.HORIZON)
        phys_indices = rollout_indices + Config.WARMUP_STEPS + 5
        scan_keys = jax.random.split(key, Config.HORIZON)
        
        # We also pass a running observation state as a carry: start with zeros for the perceptual feedback
        B = Config.BATCH_SIZE
        initial_weighted_belief = jnp.zeros((B, 4))
        
        # Convert SLAM pose from 10m space → hornet physical metres for Instar routing
        # slam_pose[:2] = (x, y) in 10m space;  slam_pose[2] = heading (same units)
        slam_xy_hornet = (slam_pose[:2] - _SLAM_OFFSET_TRAIN) / env._slam_scale
        slam_pose_hornet = jnp.concatenate([slam_xy_hornet, slam_pose[2:3]])  # (3,)
        frozen_pose  = jnp.broadcast_to(slam_pose_hornet, (B, 3))    # same pose for all agents
        frozen_surp  = jnp.full((B,), slam_surprise)                  # same surprise for all agents
        
        scan_inputs = (rollout_indices, phys_indices, scan_keys)
        init_carry = (start_state, initial_weighted_belief)

        def scan_fn(carry, xs): 
            curr_full, curr_weighted_belief = carry
            r_idx, p_idx, step_key = xs
            
            curr_robot = curr_full[0]
            wrapped_theta = jnp.mod(curr_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
            obs_robot = curr_robot.at[:, 2].set(wrapped_theta)

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

            # 3. Policy Inference
            batched_network = jax.vmap(ac_model.apply)
            mods, u_brain, _ = batched_network(params, combined_obs, action_noise)
            
            # --- LYAPUNOV HOVER & ATTENTION GATING (DNAG) ---
            # Use real SLAM surprise (frozen for this 32-step horizon).
            # Floor with positional distance so DNAG engages even before SLAM warms up.
            dist_floor = jnp.clip(
                jnp.sqrt(jnp.sum((curr_robot[:, :2] - Config.TARGET_STATE[:2])**2, axis=-1) + 1e-8) * 2.0,
                0.0, 1.0
            )
            sim_surprise = jnp.maximum(frozen_surp, dist_floor)

            # Hover modulations: dedicated hover specialist (trained ICNN + BiologicalKinematicMap).
            # hover_fixed_params is closed over and treated as a constant by JAX JIT —
            # it NEVER receives gradients from the SHAC loss.
            # noisy_obs is 8D (physical state only), matching the hover specialist's obs space.
            if _use_hover_specialist:
                hover_mods, _, _ = batched_hover_network(
                    hover_fixed_params, noisy_obs, jnp.zeros((B, 4))
                )
            else:
                # Fallback: velocity-zeroed policy (if hover_params.pkl was not found)
                hover_robot   = curr_robot.at[:, 4:8].set(0.0)
                hover_wrapped = jnp.mod(hover_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
                hover_obs_raw = hover_robot.at[:, 2].set(hover_wrapped)
                hover_obs     = jnp.concatenate([symlog(hover_obs_raw), curr_weighted_belief], axis=-1)
                hover_mods, _, _ = batched_network(params, hover_obs, jnp.zeros((B, 4)))

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
            # CSNN/STDP placeholders: replaced by real SLAM visual features in a future pass.
            key_stdp, _ = jax.random.split(key_step)
            pose_belief = frozen_pose + jax.random.normal(key_step, shape=(B, 3)) * 0.01

            norm_csnn = jnp.zeros((B, 256))  # real CSNN from SLAM debug_gates['Debug_Input_CSNN']
            norm_stdp  = jnp.zeros((B, 256))  # real STDP from SLAM debug_gates['Debug_Input_STDP']
            visual_features = (norm_csnn, norm_stdp)
            
            # Update Instar Weights & Project weighted belief for the next control step
            next_full, next_weighted_belief = env.ingest_perceptual_streams(
                next_full, pose_belief, visual_features, u_brain
            )
            
            # 5. Reward Calculation
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain, pbt_weights)

            # --- OBSTACLE COLLISION PENALTY (differentiable gradient signal) ---
            # Convert physical (x, z) → SLAM space and compute signed distance to obstacles.
            # Positive = outside all obstacles, Negative = penetrating (inside).
            _slam_sc  = env._slam_scale
            _slam_off = 5.0
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
            obs_penalty   = -Config.OBS_PENALTY_WEIGHT * obs_violation           # (B,)

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
            return (next_full, next_weighted_belief), step_metrics

        (final_full, final_weighted_belief), step_results = jax.lax.scan(
            scan_fn, init_carry, scan_inputs
        )
        
        (losses, rewards_scaled, f_errs, real_rews, m_pos, 
         m_ang_th, m_ang_ab, m_vel_lin, m_vel_ang) = step_results

        warmup_mask = rollout_indices >= Config.WARMUP_STEPS
        losses = jnp.where(warmup_mask[:, None], losses, 0.0)

        discounts = Config.GAMMA ** jnp.arange(Config.HORIZON)
        weighted_loss = jnp.dot(discounts, losses)
        
        final_robot = final_full[0]
        f_wrapped_th = jnp.mod(final_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        final_obs = symlog(final_robot.at[:, 2].set(f_wrapped_th))
        
        # Concat final observation for the critic bootstrap
        final_combined_obs = jnp.concatenate([final_obs, final_weighted_belief], axis=-1)

        _, _, final_val_actor = jax.vmap(ac_model.apply)(params, final_combined_obs)
        final_val_actor = jnp.squeeze(final_val_actor)
        
        actor_term = jnp.mean(weighted_loss - (Config.GAMMA**Config.HORIZON * final_val_actor))

        final_val_target = jax.lax.stop_gradient(final_val_actor)
        discounted_return = jnp.dot(discounts, rewards_scaled) + (Config.GAMMA**Config.HORIZON * final_val_target)
        
        start_robot = start_state[0]
        s_wrapped_th = jnp.mod(start_robot[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
        start_obs = symlog(start_robot.at[:, 2].set(s_wrapped_th))
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
    _SLAM_OFFSET_TRAIN = 5.0

    @jax.jit
    def update(params, opt_state, full_state, pbt_state, key, slam_pose, slam_surprise, obstacles):
        key_loss, key_next = jax.random.split(key)
        
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, full_state, pbt_state.weights, key_loss, slam_pose, slam_surprise, obstacles
        )
        
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        current_score = -1.0 * (logs['pos_per_agent'] * 100.0)

        new_running = 0.8 * pbt_state.running_reward + 0.2 * current_score
        new_pbt_state = pbt_state._replace(running_reward=new_running)

        return new_params, new_opt, loss, logs, next_state, new_pbt_state, key_next

    print(f"=== Starting Training: Step {start_step} to {Config.TOTAL_UPDATES} ===")
    rng, key_reset = jax.random.split(rng)
    curr_state = env.reset(key_reset, Config.BATCH_SIZE)
    
    # --- Initialise SLAM System ---
    # One SNNSLAMSystem tracks agent-0's trajectory and provides real pose + surprise
    # to the DNAG gate.  It runs OUTSIDE the JAX JIT loop (Python-native).
    print("---> Initialising SNNSLAMSystem (agent-0 tracker)...")
    slam_system = SNNSLAMSystem(jax.random.PRNGKey(7), n_depth=N_DEPTH)
    slam_system.reset(1)

    # Bootstrap SLAM from agent-0's initial pose (centre of 10m room)
    init_robot0  = np.array(curr_state[0][0])            # (8,) hornet state
    init_slam_x  = float(init_robot0[0]) * env._slam_scale + 5.0
    init_slam_z  = float(init_robot0[1]) * env._slam_scale + 5.0
    init_slam_th = float(init_robot0[2])
    slam_system.initialize_from_gt(
        jnp.array([[init_slam_x, init_slam_z]]),
        jnp.array([init_slam_th]),
    )

    # SLAM state carried between outer loop iterations
    slam_pose     = jnp.array([init_slam_x, init_slam_z, init_slam_th])  # (3,)
    slam_surprise = 0.0                                                    # float
    slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)                  # event delta baseline
    print(f"    -> SLAM initialised at ({init_slam_x:.2f}, {init_slam_z:.2f}) m, heading {init_slam_th:.2f} rad")
    
    # Helper: regenerate arena + reset SLAM at episode boundaries
    def _reset_slam_for_new_arena(key, curr_agent0_state):
        """
        Called at forced-reset boundaries (PBT + periodic RESET_INTERVAL).
        1. Regenerates obstacle room (new episode = new room).
        2. Resets SNNSLAMSystem and re-initialises from agent-0's current pose.
        3. Clears the event-camera intensity baseline.
        Returns updated slam_pose, slam_surprise, slam_prev_int.
        """
        nonlocal slam_prev_int
        env.regenerate_arena(key=key)

        # Re-initialise SLAM from agent-0's current pose in the new room
        r0 = np.array(curr_agent0_state)
        new_slam_x  = float(r0[0]) * env._slam_scale + 5.0
        new_slam_z  = float(r0[1]) * env._slam_scale + 5.0
        new_slam_th = float(r0[2])

        slam_system.reset(1)
        slam_system.initialize_from_gt(
            jnp.array([[new_slam_x, new_slam_z]]),
            jnp.array([new_slam_th]),
        )
        slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)  # clear event baseline

        new_slam_pose = jnp.array([new_slam_x, new_slam_z, new_slam_th])
        print(f"    ---> New arena + SLAM reset: pose ({new_slam_x:.2f}, {new_slam_z:.2f}) m")
        return new_slam_pose, 0.0  # fresh room → surprise starts at 0

    # --- JIT Compilation ---
    print("---> Compiling JAX Update...")
    t0 = time.time()
    key_compile = jax.random.PRNGKey(0)
    _ = update(params, opt_state, curr_state, pbt_state, key_compile, slam_pose, jnp.array(slam_surprise), env._obstacles)
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
            slam_pose, jnp.array(slam_surprise), env._obstacles
        )

        # 1b. SLAM update (outside JIT, runs on agent-0's terminal state)
        # Compute sensor bundle from the final state of this horizon
        robot0_np = np.array(next_state[0][0])  # (8,) agent-0 hornet state
        ev_jax, kin_jax, tof_jax, slam_prev_int = env.compute_slam_sensors(
            robot0_np, slam_prev_int
        )
        # Run one SLAM step (closed-loop with full memory + loop closure detection)
        try:
            pose_est, _, _, _, _, debug_gates = slam_system.forward_step(
                ev_jax, kin_jax, tof_jax,
                inject_drift=False, autopilot_on=(slam_surprise < 0.60)
            )
            slam_pose     = jnp.array([
                float(pose_est[0, 0]),   # x in 10m space
                float(pose_est[0, 1]),   # y in 10m space
                float(pose_est[0, 2]),   # heading
            ])
            slam_surprise = float(1.0 - float(debug_gates['Raw_Match'][0]))
        except Exception as _slam_err:
            # Graceful fallback: keep previous SLAM values
            print(f"    [SLAM] non-fatal error at step {i}: {_slam_err}")
    
        # 2. Stability Checks
        if jnp.isnan(loss):
            print(f"!!! CRITICAL: NaN detected at step {i} !!! Rolling back params + resetting batch.")
            params    = _prev_params     # restore last-known-good weights
            opt_state = _prev_opt_state  # restore last-known-good optimizer state
            rng, k_res = jax.random.split(rng)
            curr_state = env.reset(k_res, Config.BATCH_SIZE)
            continue
        
        r_state = next_state[0]
        
        # --- Environment Reset Logic ---
        is_nan = jnp.isnan(r_state).any(axis=1)
        is_oor = (jnp.abs(r_state[:, 0]) > Config.ARENA_W) | (jnp.abs(r_state[:, 1]) > Config.ARENA_W)
        # Obstacle collision: check in SLAM space using numpy (fast, outside JIT)
        _obs_np     = np.array(env._obstacles) if env._obstacles is not None else np.zeros((0, 4))
        _slam_xs_np = np.array(r_state[:, 0]) * float(env._slam_scale) + 5.0
        _slam_zs_np = np.array(r_state[:, 1]) * float(env._slam_scale) + 5.0
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
            slam_pose, slam_surprise = _reset_slam_for_new_arena(
                k_arena, curr_state[0][0]  # agent-0's state before reset
            )
            curr_state = env.reset(rng, Config.BATCH_SIZE)

        print(f"Step {i:04d} | Epoch: {dt_epoch:.2f}s | Total: {total_elapsed/60:.1f}min | "
              f"Loss: {loss:.1e} (A:{logs['act_loss']:.1e} C:{logs['crit_loss']:.1e}) | "
              f"Rew: {logs['rew']:.1f}\n"
              f"    -> Errs[Pos:{logs['pos']:.2f} Th:{logs['ang_th']:.2f} Ab:{logs['ang_ab']:.2f} "
              f"LVel:{logs['vel_lin']:.2f} AVel:{logs['vel_ang']:.2f} Frc:{logs['ferr']:.4f}] | "
              f"MeanPos: [{mean_x:+.2f}, {mean_z:+.2f}]\n"
              f"    -> Phys: [Hum:{raw_hum_energy:.0f} | ThVel:{thorax_mag:.2f} rad/s]")
    
        if i % Config.VIS_INTERVAL == 0:
            ckpt_path = os.path.join(Config.CKPT_DIR, f"shac_params_{i}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    'params': params, 
                    'opt_state': opt_state,
                    'pbt_state': pbt_state 
                }, f)
            print(f"--> Saved Checkpoint: {ckpt_path}")
            run_visualization(env, params, i, vis_step_fn)

# Global Visualization Step
@partial(jax.jit, static_argnums=(0,))
def vis_step_fn(env, curr_state, curr_params, step_idx):
    r_st = curr_state[0]
    B = r_st.shape[0]
    
    # 1. Prepare Observation
    wrapped_th = jnp.mod(r_st[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
    obs_v = r_st.at[:, 2].set(wrapped_th)
    scaled_obs = symlog(obs_v)
    
    # 2. Ingest visual-spatial belief features
    pose_belief = r_st[:, :3]
    norm_csnn = jnp.zeros((B, 256)).at[:, 0].set(1.0)
    norm_stdp = jnp.zeros((B, 256)).at[:, 0].set(1.0)
    visual_features = (norm_csnn, norm_stdp)
    
    next_state, weighted_belief = env.ingest_perceptual_streams(
        curr_state, pose_belief, visual_features, jnp.zeros((B, 4))
    )
    
    combined_obs = jnp.concatenate([scaled_obs, weighted_belief], axis=-1)
    
    # 3. Policy Inference
    mods, _, _ = ac_model.apply(curr_params, combined_obs)
    
    # 4. Env Step
    next_state, _, f_nodal, w_pose, h_marker = env.step_batch(next_state, mods, step_idx=step_idx)
    
    return next_state, f_nodal, w_pose, h_marker

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