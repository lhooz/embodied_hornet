import os
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
from .neural_idapbc import policy_network_icnn, hover_stable, differentiable_attention_gate, unpack_action, ScaleConfig
from .env import FlyEnv

# --- BASE MODULES FROM hornetRL SIBLING REPO ---
from hornetRL.fluid_surrogate import JaxSurrogateEngine
from hornetRL.fly_system import FlappingFlySystem, PhysParams
from hornetRL.neural_cpg import OscillatorState, step_oscillator, get_wing_kinematics
from hornetRL.pbt_manager import init_pbt_state, pbt_evolve

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
    ARENA_W = 0.45 

    # --- Time Scales ---
    DT = 3e-5               # Physics integration timestep (s)
    SIM_SUBSTEPS = 72       # Physics steps per Control Step
                            
    HORIZON = 32            # Trajectory horizon for Backpropagation (= 8 wingbeats @ 4 steps/beat)
    RESET_INTERVAL = 50     
    PBT_INTERVAL = 500      

    BATCH_SIZE = 32          
    LR_ACTOR = 5e-4         
    MAX_GRAD_NORM = 1.0     
    GAMMA = 0.99            

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
# 3. VISUALIZATION ENGINE
# ==============================================================================
def run_visualization(env, params, update_idx, vis_step_fn):
    print(f"--> Generating Visualization for Step {update_idx}...")

    steps_per_frame = 1
    total_visual_frames = Config.HORIZON * 8

    sim_data = {'states': [], 'wing_pose': [], 'nodal_forces': [], 'le_marker': [], 'hinge_marker': [], 't': []}
    
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

        sim_data['states'].append(np.array(state[0][0])) 
        sim_data['t'].append(current_step_counter * Config.DT)
        
        f_st = state[1]
        sim_data['le_marker'].append(np.array(f_st.marker_le[0]))
        sim_data['wing_pose'].append(np.array(w_pose[0]))
        sim_data['nodal_forces'].append(np.array(f_nodal[0]))
        sim_data['hinge_marker'].append(np.array(h_marker[0]))

    # --- Matplotlib Animation Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    patch_thorax = patches.Ellipse((0,0), linewidth=1.0, width=0.012, height=0.006, facecolor='#333333', edgecolor='black', zorder=10)
    patch_head = patches.Circle((0,0), linewidth=1.0, radius=0.0025, facecolor='#00FF00', edgecolor='black', zorder=10)
    patch_abd = patches.Ellipse((0,0), linewidth=1.0, width=0.018, height=0.008, facecolor='#1f77b4', edgecolor='black', alpha=0.8, zorder=9)
    ax.add_patch(patch_thorax)
    ax.add_patch(patch_head)
    ax.add_patch(patch_abd)

    real_line, = ax.plot([], [], 'k-', linewidth=1.0, alpha=0.8, zorder=12)
    patch_le = patches.Circle((0,0), radius=0.001, color='red', zorder=15)
    ax.add_patch(patch_le)

    patch_hinge = patches.Circle((0,0), radius=0.001, color='orange', zorder=15)
    ax.add_patch(patch_hinge)

    dummy = np.zeros(20)
    quiver = ax.quiver(dummy, dummy, dummy, dummy, color='red', scale=3.0, scale_units='xy', zorder=20, width=0.0002)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')
    ax.grid(True, linestyle=':', alpha=0.3)

    def update(frame):
        if frame >= len(sim_data['states']): return
        
        r_state = sim_data['states'][frame]
        w_pose = sim_data['wing_pose'][frame]
        f_nodal = sim_data['nodal_forces'][frame]
        le_pos = sim_data['le_marker'][frame]
        hinge_pos = sim_data['hinge_marker'][frame]
        t = sim_data['t'][frame]
        
        rx, rz = r_state[0], r_state[1]
        r_th, r_phi = r_state[2], r_state[3]
        
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        
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
        N_pts = env.phys.fluid.N_PTS
        x_local = np.linspace(wing_len/2, -wing_len/2, N_pts)
        c_w, s_w = np.cos(wang), np.sin(wang)
        wing_x = wx + x_local * c_w
        wing_z = wz + x_local * s_w
        real_line.set_data(wing_x, wing_z)
        
        patch_le.set_center((le_pos[0], le_pos[1]))
        patch_hinge.set_center((hinge_pos[0], hinge_pos[1]))
        
        pts = np.stack([wing_x, wing_z], axis=1)
        quiver.set_offsets(pts)
        quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
        
        time_text.set_text(f"T: {t:.4f}s | Y: {rz:.3f}")
        return patch_thorax, patch_le, patch_hinge

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
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )
        opt_state = optimizer.init(params)
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    else:
        print("--> [SCRATCH] No checkpoint or expert found. Initializing random population.")
        single_params = ac_model.init(rng, dummy_input)
        params = jax.tree.map(lambda x: jnp.stack([x] * Config.BATCH_SIZE), single_params)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
            optax.adam(Config.LR_ACTOR)
        )
        opt_state = optimizer.init(params)
        pbt_state = init_pbt_state(rng, Config.BATCH_SIZE, Config.PBT_BASE_WEIGHTS)

    print(f"--> Initialization Complete. Params Batch Shape: {params['linear']['w'].shape}")

    def loss_fn(params, start_state, pbt_weights, key):
        """
        Computes the total loss over the trajectory horizon.
        Includes policy gradient, value function loss, and auxiliary force matching loss.
        """
        rollout_indices = jnp.arange(Config.HORIZON)
        phys_indices = rollout_indices + Config.WARMUP_STEPS + 5
        scan_keys = jax.random.split(key, Config.HORIZON)
        
        # We also pass a running observation state as a carry: start with zeros for the perceptual feedback
        B = Config.BATCH_SIZE
        initial_weighted_belief = jnp.zeros((B, 4))
        # Carry previous visual features for visual-similarity-based surprise (matches SLAM's 1−Raw_Match)
        initial_prev_visual = jnp.zeros((B, 256))
        
        scan_inputs = (rollout_indices, phys_indices, scan_keys)
        init_carry = (start_state, initial_weighted_belief, initial_prev_visual)

        def scan_fn(carry, xs): 
            curr_full, curr_weighted_belief, prev_visual = carry
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
            # Visual-similarity-based Surprise (matches SLAM's S = 1.0 − Raw_Match)
            # Compute cosine similarity between consecutive CSNN frames
            vis_base = curr_robot[:, :2]
            key_vis, key_step_rest = jax.random.split(step_key)
            curr_visual = jax.random.normal(key_vis, shape=(B, 256)) * 0.05
            curr_visual = curr_visual.at[:, :2].add(vis_base)
            curr_visual = curr_visual / (jnp.linalg.norm(curr_visual, axis=-1, keepdims=True) + 1e-8)
            
            # Cosine similarity → surprise (1.0 = completely novel, 0.0 = perfect match)
            cos_sim = jnp.sum(prev_visual * curr_visual, axis=-1)  # (B,)
            vis_surprise = jnp.clip(1.0 - cos_sim, 0.0, 1.0)
            # Floor: use positional distance as a lower bound to bootstrap when prev_visual is zeros
            dist_floor = jnp.clip(jnp.sqrt(jnp.sum((curr_robot[:, :2] - Config.TARGET_STATE[:2])**2, axis=-1) + 1e-8) * 2.0, 0.0, 1.0)
            sim_surprise = jnp.maximum(vis_surprise, dist_floor)
            
            # Brain-to-muscle mapping for active IDA-PBC Hover stabilization
            hover_mods, _ = jax.vmap(hover_stable)(noisy_obs)
            
            # Fully differentiable attention gate blending
            blended_mods, alpha = jax.vmap(differentiable_attention_gate)(sim_surprise, mods, hover_mods)
            
            # 4. Environment Step (Physics uses the blended passivity-preserving actions)
            next_full, f_actual, _, _, _ = env.step_batch(curr_full, blended_mods, step_idx=p_idx)
            
            # --- SIMULATE PERCEPTUAL STREAM INGESTION (460 Hz) ---
            # Reuse the visual frame computed for surprise (CSNN) and generate STDP
            norm_csnn = curr_visual  # Already computed above for surprise

            key_stdp, _ = jax.random.split(key_step_rest)
            next_robot = next_full[0]
            vis_base_next = next_robot[:, :2]
            norm_stdp = jax.random.normal(key_stdp, shape=(B, 256)) * 0.05
            norm_stdp = norm_stdp.at[:, :2].add(vis_base_next)
            norm_stdp = norm_stdp / (jnp.linalg.norm(norm_stdp, axis=-1, keepdims=True) + 1e-8)
            visual_features = (norm_csnn, norm_stdp)
            
            # Route believed pose
            pose_belief = curr_robot[:, :3] + jax.random.normal(key_step, shape=(B, 3)) * 0.005
            
            # Update Instar Weights & Project weighted belief for the next control step
            next_full, next_weighted_belief = env.ingest_perceptual_streams(
                next_full, pose_belief, visual_features, u_brain
            )
            
            # 5. Reward Calculation
            rew_scaled, met = env.get_reward_metrics(curr_robot, u_brain, pbt_weights)
            
            # --- SCALE Force ---
            raw_diff = u_brain[:, :3] - f_actual[:, :3]
            norm_diff = raw_diff / Config.FORCE_NORMALIZER[:3]
            force_err_norm = jnp.mean(jnp.square(symlog(norm_diff)))
            
            # Apply Loss Weight
            loss_t = -rew_scaled + (Config.AUX_LOSS_WEIGHT * force_err_norm)
            force_err_raw = jnp.mean(raw_diff**2)
            
            step_metrics = (
                loss_t, rew_scaled, force_err_raw, 
                met['rew'], met['pos'], 
                met['ang_th'], met['ang_ab'], 
                met['vel_lin'], met['vel_ang']
            )
            return (next_full, next_weighted_belief, curr_visual), step_metrics

        (final_full, final_weighted_belief, _final_visual), step_results = jax.lax.scan(
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

    @jax.jit
    def update(params, opt_state, full_state, pbt_state, key): 
        key_loss, key_next = jax.random.split(key)
        
        (loss, (logs, next_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, full_state, pbt_state.weights, key_loss
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
    
    # --- JIT Compilation ---
    print("--> Compiling JAX Update...")
    t0 = time.time()
    key_compile = jax.random.PRNGKey(0)
    _ = update(params, opt_state, curr_state, pbt_state, key_compile) 
    print(f"--> Compilation Finished in {time.time() - t0:.2f}s")

    # --- Main Loop ---
    key_explore = jax.random.PRNGKey(999) 
    t_start = time.time()
    
    for i in range(start_step, Config.TOTAL_UPDATES):
        t0 = time.time() 

        if i == start_step:
            print(f"DEBUG: Env Target Theta: {env.target[2]:.4f}")
            print(f"DEBUG: Spawn Theta: {curr_state[0][0, 2]:.4f}")

        # 1. Update Step
        params, opt_state, loss, logs, next_state, pbt_state, key_explore = update(
            params, opt_state, curr_state, pbt_state, key_explore
        )
    
        # 2. Stability Checks
        if jnp.isnan(loss):
            print(f"!!! CRITICAL: NaN detected at step {i} !!! Reseting batch.")
            rng, k_res = jax.random.split(rng)
            curr_state = env.reset(k_res, Config.BATCH_SIZE)
            continue
        
        r_state = next_state[0]
        
        # --- Environment Reset Logic ---
        is_nan = jnp.isnan(r_state).any(axis=1)
        is_crashed = (jnp.abs(r_state[:, 0]) > Config.ARENA_W) | (jnp.abs(r_state[:, 1]) > Config.ARENA_W)
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
            print(f"--> PBT EVOLUTION (Step {i})")
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

        # --- B. FORCED RESET ---
        if (i % Config.RESET_INTERVAL == 0 and i > 0) or performed_pbt:
            if performed_pbt:
                print("    -> PBT Mutation applied. Forcing Environment Reset.")
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