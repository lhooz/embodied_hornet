import sys
import os
sys.path.append("/Users/lhooz/.openclaw/workspace/embodied_hornet")

import jax
import jax.numpy as jnp
import numpy as np
import pickle

import embodied_hornet.train as train_mod

Config = train_mod.Config
env = train_mod.FlyEnv(Config)

_hover_pkl = "/Users/lhooz/.openclaw/workspace/embodied_hornet/embodied_hornet/hover_params.pkl"
with open(_hover_pkl, "rb") as f:
    hover_ckpt = pickle.load(f)
hover_params = jax.tree.map(jnp.array, hover_ckpt['params'])
params_single = jax.tree.map(lambda x: x[0], hover_params)

state = env.reset(jax.random.PRNGKey(42), 1)
# Start at center with target at (0.0, 0.0)
r_state_override = state[0].at[0].set(Config.TARGET_STATE)
state = (r_state_override,) + state[1:]

target_xy = jnp.array([[0.5, -0.5]])

print("Testing with target at (0.5, -0.5) to see if it saturates:")

for step in range(5):
    r_st = state[0]
    slam_pose_jax = jnp.array([[r_st[0, 0] + 1.0, r_st[0, 1] + 1.0, r_st[0, 2] - 1.0]])
    slam_xy_hornet = (slam_pose_jax[:, :2] - 1.0) / env._slam_scale
    pose_belief = jnp.concatenate([slam_xy_hornet, slam_pose_jax[:, 2:3]], axis=-1)
    
    obs_v = r_st
    obs_v = obs_v.at[:, 0].set(pose_belief[:, 0] - target_xy[:, 0])
    obs_v = obs_v.at[:, 1].set(pose_belief[:, 1] - target_xy[:, 1])
    body_pitch_est = pose_belief[:, 2] + 1.0
    wrapped_th = jnp.mod(body_pitch_est + jnp.pi, 2 * jnp.pi) - jnp.pi
    obs_v = obs_v.at[:, 2].set(wrapped_th)
    scaled_obs = train_mod.symlog(obs_v)
    
    # Run policy
    hover_mods, u_brain, _ = train_mod.hover_ac_model.apply(
        params_single, scaled_obs, None, None, 0.0, None, 0.0, 0.0, None
    )
    
    print(f"Step {step:2d} | Pose: x={r_st[0,0]:.4f}, z={r_st[0,1]:.4f}, th={r_st[0,2]:.4f}")
    print(f"        | Obs_rel: x={obs_v[0,0]:.4f}, z={obs_v[0,1]:.4f}, th={obs_v[0,2]:.4f}")
    print(f"        | Brain forces: Fx={u_brain[0,0]:.6f}, Fz={u_brain[0,1]:.6f}, Tau_th={u_brain[0,2]:.6f}")
    print(f"        | Action mods : d_freq={hover_mods[0,0]:.4f}, d_amp={hover_mods[0,1]:.4f}, bias_target={hover_mods[0,2]:.6f}, pitch_off={hover_mods[0,3]:.4f}")
    
    state, _, _, _, _ = env.step_batch(state, hover_mods, step_idx=step+5)
