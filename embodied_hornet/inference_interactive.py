import os
import sys
import platform

# Force CPU for local execution unless --gpu is explicitly specified
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ["JAX_PLATFORMS"] = "cpu"
elif "--gpu" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import time
import glob
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

import jax
import jax.numpy as jnp
import haiku as hk

# --- INTEGRATION MODULES ---
from .train import (
    ac_model, hover_ac_model, vis_step_fn, symlog, Config, _get_ray_indices
)
from .env import FlyEnv
from .neural_idapbc import ScaleConfig
from .reichardt_emd import N_EMD_PIX

# --- BASE MODULES FROM hornetRL SIBLING REPO ---
from hornetRL.pbt_manager import init_pbt_state
from snn_slam_system import SNNSLAMSystem, N_DEPTH, SpikingOccupancyGrid
from sparse_forest import N_PIXELS, compute_tof_distance, compute_pixel_readings, THRESHOLD

def get_latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "shac_params_*.pkl"))
    if not files:
        return None
    epochs = []
    for f in files:
        m = re.search(r"shac_params_(\d+)\.pkl", f)
        if m:
            epochs.append((int(m.group(1)), f))
    if not epochs:
        return None
    epochs.sort(key=lambda x: x[0])
    return epochs[-1][1]

class InteractiveInference:
    def __init__(self, checkpoint_path, save_gif_path, draw_skip=5, max_steps=None):
        self.checkpoint_path = checkpoint_path
        self.save_gif_path = save_gif_path
        self.draw_skip = draw_skip
        
        # Check if running headless with no steps set
        is_headless = plt.get_backend().lower() == 'agg'
        if is_headless and max_steps is None:
            max_steps = 50
            print("⚠️ [HEADLESS DETECTED] Defaulting to 50 steps to prevent hanging.")
        self.max_steps = max_steps
        
        self.env = FlyEnv(Config)
        # Randomize the arena obstacles/layout upon startup
        import time
        rand_seed = int(time.time() * 1000) % 1000000
        self.env.regenerate_arena(seed=rand_seed, quiet=True)
        print(f"--> Randomized Inference Arena (seed={rand_seed})")
        self.rng = jax.random.PRNGKey(Config.SEED)
        
        # Load checkpoint
        print(f"--> Loading Checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        params = checkpoint['params']
        pbt_state = checkpoint.get('pbt_state', None)
        
        # Determine best agent
        self.best_idx = 0
        if pbt_state is not None and hasattr(pbt_state, 'running_reward'):
            self.best_idx = int(np.argmax(pbt_state.running_reward))
            print(f"--> Best Agent index: {self.best_idx} (reward: {pbt_state.running_reward[self.best_idx]:.2f})")
        else:
            print("--> No PBT running reward found. Defaulting to agent 0.")
            
        self.params_single = jax.tree.map(lambda x: x[self.best_idx], params)
        
        # Load hover specialist
        _hover_pkl = os.path.join(os.path.dirname(__file__), "hover_params.pkl")
        self.hover_params_single = None
        if os.path.exists(_hover_pkl):
            with open(_hover_pkl, "rb") as f:
                _hover_ckpt = pickle.load(f)
            hover_fixed_params = jax.tree.map(jnp.array, _hover_ckpt['params'])
            self.hover_params_single = jax.tree.map(lambda x: x[0], hover_fixed_params)
            print(f"--> Loaded hover specialist from {_hover_pkl}")
        else:
            print("--> hover_params.pkl not found — using velocity-zeroed fallback.")
            
        # Initialize state
        self.rng, key_reset = jax.random.split(self.rng)
        self.state = self.env.reset(key_reset, 1)
        
        # Position override: start at target hover state
        r_state_override = self.state[0].at[0].set(Config.TARGET_STATE)
        self.state = (r_state_override,) + self.state[1:]
        
        # Initialize SLAM
        self.vis_slam = SNNSLAMSystem(jax.random.PRNGKey(999), n_depth=N_DEPTH)
        self.vis_slam.reset(1)
        
        r_state_np_start = np.array(self.state[0][0])
        self.start_slam_x = float(r_state_np_start[0]) * self.env._slam_scale + 1.0
        self.start_slam_z = float(r_state_np_start[1]) * self.env._slam_scale + 1.0
        self.start_slam_th = float(r_state_np_start[2]) - 1.0
        
        self.env._prev_robot_state = None
        self.vis_slam.initialize_pose(
            jnp.array([[self.start_slam_x, self.start_slam_z]]),
            jnp.array([self.start_slam_th]),
        )
        
        # Initialize SOG
        self._sog = SpikingOccupancyGrid(map_size_m=2.0, res=0.04, offset_m=0.0, v_max=1.0)
        self._sog_state = self._sog.init_state()
        
        # Dynamic target
        self.vis_target_xy = jnp.array([[0.5, -0.5]]) # start target at (0.5, -0.5) physical
        
        # Interactive flags
        self.is_running = True
        
        # Trajectory recording
        self.sim_data = {
            'states': [], 'wing_pose': [], 'nodal_forces': [],
            'le_marker': [], 'hinge_marker': [], 't': [],
            'slam_pos': [], 'slam_est': [], 'imu_dr': [],
            'surprise': [], 'tof': [], 'heading': [], 'events': [],
            'active_places': [], 'alpha': [], 'f_repel': [],
            'flow_corr': [], 'f_repel_vec': [], 'f_brain_vec': [],
            'f_net_vec': [], 'f_emd_vec': [], 'f_instar_vec': [],
            'sog_states': [], 'target_xy': [], # track dynamic targets
            'smooth_omega': [], 'learn_gate': []
        }
        self.sim_data['sog_res'] = self._sog.res
        self.sim_data['sog_grid_w'] = self._sog.grid_w
        
    def setup_plots(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.patch.set_facecolor('#0d0d0d')
        gs = self.fig.add_gridspec(2, 3)
        
        self.ax_nav = self.fig.add_subplot(gs[0, 0])
        self.ax_map = self.fig.add_subplot(gs[1, 0])
        self.ax_wing = self.fig.add_subplot(gs[0, 1])
        self.ax_snn = self.fig.add_subplot(gs[1, 1])
        self.ax_telemetry = self.fig.add_subplot(gs[0, 2])
        self.ax_vis = self.fig.add_subplot(gs[1, 2])
        
        for ax in (self.ax_nav, self.ax_map, self.ax_wing, self.ax_snn, self.ax_telemetry, self.ax_vis):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
                
        # --- Navigation Room ---
        self.ax_nav.set_xlim(0, 2)
        self.ax_nav.set_ylim(0, 2)
        self.ax_nav.set_aspect('equal')
        self.ax_nav.set_title('Navigation Room (SLAM Space) - CLICK TO ASSIGN TARGET', color='white', fontsize=10, fontweight='bold')
        self.ax_nav.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
        self.ax_nav.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)
        
        room_rect = plt.Rectangle((0, 0), 2, 2, linewidth=2, edgecolor='#00ffcc', facecolor='none')
        self.ax_nav.add_patch(room_rect)
        
        obstacles_np = np.array(self.env._obstacles) if self.env._obstacles is not None else np.zeros((0, 4))
        self.obs_patch_list = []
        for obs in obstacles_np:
            x0, y0, x1, y1 = obs
            p = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, zorder=4)
            self.ax_nav.add_patch(p)
            self.obs_patch_list.append((obs, p))
            
        # Draw dynamic target marker
        tgt_phys = np.array(self.vis_target_xy[0])
        tgt_u = tgt_phys[0] * self.env._slam_scale + 1.0
        tgt_v = tgt_phys[1] * self.env._slam_scale + 1.0
        self.tgt_nav_marker, = self.ax_nav.plot(tgt_u, tgt_v, marker='*', markersize=14, color='#ffdd00',
                                                 markeredgecolor='#ff8800', zorder=20, label='Target')
        
        # Trajectories and arrows
        self.traj_line, = self.ax_nav.plot([], [], '-', color='#00ff88', linewidth=1.0, alpha=0.6, zorder=5, label='Path (GT)')
        self.hornet_dot, = self.ax_nav.plot([], [], 'o', color='#ff4444', markersize=7, zorder=15, label='Hornet (GT)')
        self.slam_est_line, = self.ax_nav.plot([], [], ':', color='#ffa500', linewidth=1.2, alpha=0.8, zorder=6, label='Path (SLAM)')
        self.slam_est_dot, = self.ax_nav.plot([], [], 's', color='#ffa500', markersize=5, zorder=14, label='SLAM Pose')
        self.imu_dr_line, = self.ax_nav.plot([], [], '--', color='#00ffff', linewidth=0.8, alpha=0.5, zorder=4, label='Path (IMU DR)')
        self.imu_dr_dot, = self.ax_nav.plot([], [], '^', color='#00ffff', markersize=5, zorder=13, label='IMU DR Pose')
        
        self.heading_arr = self.ax_nav.quiver([0], [0], [0], [0], color='#ff8888', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Sensor Heading')
        self.slam_heading_arr = self.ax_nav.quiver([0], [0], [0], [0], color='#ffa500', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=15, label='SLAM Heading')
        self.vel_arr = self.ax_nav.quiver([0], [0], [0], [0], color='#ff33ff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Velocity')
        self.target_dir_arr = self.ax_nav.quiver([0], [0], [0], [0], color='#ffcc00', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Believed Target Dir')
        
        # ToF Beams
        self.tof_beam_artists = []
        _beam_colours = ['#00aaff', '#ffff44', '#00aaff', '#ff5555']
        for _bc in _beam_colours:
            _bl, = self.ax_nav.plot([], [], '-',  color=_bc, linewidth=1.5, alpha=0.85, zorder=12)
            _bm, = self.ax_nav.plot([], [], 'D',  color=_bc, markersize=4,  alpha=0.95, zorder=13)
            self.tof_beam_artists.append((_bl, _bm))
            
        # Camera FOV boundary
        self.fov_left_line,  = self.ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
        self.fov_right_line, = self.ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
        
        self.nav_time = self.ax_nav.text(0.02, 0.97, '', transform=self.ax_nav.transAxes,
                                          color='#cccccc', fontsize=8, va='top', family='monospace')
        self.ax_nav.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)
        
        # 1D Event camera strip
        self.ax_vis.set_xticks([])
        self.ax_vis.set_yticks([])
        self.ax_vis.set_title('1D Event Camera Stream (Green=ON, Red=OFF)', color='white', fontsize=10, fontweight='bold')
        colors_ev = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.8, 0.1)]
        cm_ev = LinearSegmentedColormap.from_list('events_cmap', colors_ev, N=3)
        self.vis_strip = self.ax_vis.imshow(np.zeros((1, N_PIXELS)), cmap=cm_ev, vmin=-1.0, vmax=1.0, aspect='auto')
        
        # --- SOG Heatmap ---
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title('Spiking Occupancy Grid (Robot Memory)', color='white', fontsize=10, fontweight='bold')
        self.ax_map.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
        self.ax_map.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)
        
        SOG_EXTENT = [0, 2, 0, 2]
        self.occ_display = np.zeros((self._sog.grid_w, self._sog.grid_w), dtype=np.float32)
        self.occ_img = self.ax_map.imshow(
            self.occ_display, origin='lower', extent=SOG_EXTENT,
            cmap='magma', vmin=-0.2, vmax=1.0, aspect='equal', zorder=1
        )
        self.map_robot_dot, = self.ax_map.plot([], [], 'o', color='#00ffcc', markersize=5, zorder=10, label='Robot')
        self.map_trail_line, = self.ax_map.plot([], [], '-', color='#00ffcc', linewidth=0.8, alpha=0.4, zorder=5, label='Trail')
        
        # Add target marker to ax_map
        self.tgt_map_marker, = self.ax_map.plot(tgt_u, tgt_v, marker='*', markersize=14, color='#ffdd00',
                                                 markeredgecolor='#ff8800', zorder=20, label='Target')
        
        self.map_target_dir_arr = self.ax_map.quiver([0], [0], [0], [0], color='#ffcc00', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Believed Target Dir')
        self.map_f_repel_arr = self.ax_map.quiver([0], [0], [0], [0], color='#ff33ff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='SOG Repulsion Force')
        self.map_f_brain_arr = self.ax_map.quiver([0], [0], [0], [0], color='#ff8800', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Brain Goal Force')
        self.map_f_emd_arr = self.ax_map.quiver([0], [0], [0], [0], color='#39ff14', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='EMD Reflex Force')
        self.map_f_instar_arr = self.ax_map.quiver([0], [0], [0], [0], color='#ff007f', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Instar Memory Force')
        self.map_f_net_arr = self.ax_map.quiver([0], [0], [0], [0], color='#00ffff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=18, label='Net Control Force')
        
        self.map_tof_beam_artists = []
        for _bc in _beam_colours:
            _bl, = self.ax_map.plot([], [], '-',  color=_bc, linewidth=1.2, alpha=0.6, zorder=12)
            _bm, = self.ax_map.plot([], [], 'D',  color=_bc, markersize=3,  alpha=0.7, zorder=13)
            self.map_tof_beam_artists.append((_bl, _bm))
            
        self.map_fov_left_line,  = self.ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
        self.map_fov_right_line, = self.ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
        self.ax_map.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)
        
        # --- Wing Mechanics ---
        self.ax_wing.set_aspect('equal')
        self.ax_wing.set_title('Wing Mechanics (Close-up)', color='white', fontsize=10, fontweight='bold')
        self.ax_wing.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
        self.ax_wing.set_ylabel('Z (m)', color='#aaaaaa', fontsize=8)
        self.ax_wing.grid(True, linestyle=':', alpha=0.2, color='#444444')
        
        room_rect_wing = plt.Rectangle((-1.0, -1.0), 2.0, 2.0, linewidth=2, edgecolor='#00ffcc', facecolor='none', linestyle='--', zorder=2)
        self.ax_wing.add_patch(room_rect_wing)
        
        self.obs_patch_wing_list = []
        for obs in obstacles_np:
            px0, py0, px1, py1 = obs - 1.0
            p_wing = plt.Rectangle((px0, py0), px1 - px0, py1 - py0,
                                   facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, alpha=0.8, zorder=3)
            self.ax_wing.add_patch(p_wing)
            self.obs_patch_wing_list.append((obs - 1.0, p_wing))
            
        self.patch_thorax = patches.Ellipse((0,0), linewidth=1.0, width=0.012, height=0.006,
                                       facecolor='#555555', edgecolor='#aaaaaa', zorder=10)
        self.patch_head   = patches.Circle((0,0), linewidth=1.0, radius=0.0025,
                                       facecolor='#00FF88', edgecolor='#aaaaaa', zorder=10)
        self.patch_abd    = patches.Ellipse((0,0), linewidth=1.0, width=0.018, height=0.008,
                                       facecolor='#4488cc', edgecolor='#aaaaaa', alpha=0.8, zorder=9)
        self.ax_wing.add_patch(self.patch_thorax)
        self.ax_wing.add_patch(self.patch_head)
        self.ax_wing.add_patch(self.patch_abd)
        
        # --- WINGS (Ghost/Shadow trail to solve aliasing: 9 poses spanning 0.5 cycles) ---
        self.wing_lines = []
        shadow_alphas = np.linspace(0.05, 0.4, 8)
        all_alphas = np.concatenate([shadow_alphas, [1.0]])
        for a in all_alphas:
            lw = 1.5 if a == 1.0 else 1.0
            col = '#ffffff' if a == 1.0 else '#cccccc'
            wl, = self.ax_wing.plot([], [], col, linestyle='-', linewidth=lw, alpha=a, zorder=11 if a == 1.0 else 10)
            self.wing_lines.append(wl)
            
        self.patch_le    = patches.Circle((0,0), radius=0.001, color='#ff4444', zorder=15)
        self.patch_hinge = patches.Circle((0,0), radius=0.001, color='#ff8800', zorder=15)
        self.ax_wing.add_patch(self.patch_le)
        self.ax_wing.add_patch(self.patch_hinge)
        
        dummy = np.zeros(20)
        self.quiver = self.ax_wing.quiver(dummy, dummy, dummy, dummy, color='#ff6666',
                                          scale=3.0, scale_units='xy', zorder=20, width=0.002)
        self.time_text = self.ax_wing.text(0.02, 0.95, '', transform=self.ax_wing.transAxes, color='#cccccc', fontsize=8)
        
        # --- Place Cell Spike Raster ---
        self.snn_spike_line, = self.ax_snn.plot([], [], 'o', color='#00ffff', markersize=2, alpha=0.4, linestyle='None')
        self.ax_snn.set_xlim(0, 5.0)
        self.ax_snn.set_ylim(-5, 260)
        self.ax_snn.set_title('Place Cell Spike Raster', color='white', fontsize=10, fontweight='bold')
        self.ax_snn.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
        self.ax_snn.set_ylabel('Neuron ID (0-255)', color='#aaaaaa', fontsize=8)
        self.ax_snn.grid(True, linestyle=':', alpha=0.2, color='#444444')
        self.snn_time_line = self.ax_snn.axvline(x=0, color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)
        
        # --- Closed-Loop Telemetry Chart ---
        self.line_surprise, = self.ax_telemetry.plot([], [], '-', color='#ffa500', linewidth=1.2, label='SLAM Surprise')
        self.line_alpha, = self.ax_telemetry.plot([], [], '-', color='#00ffff', linewidth=1.2, label='DNAG Alpha (Gate)')
        
        self.ax_telemetry.set_xlim(0, 5.0)
        self.ax_telemetry.set_ylim(-0.05, 1.1)
        self.ax_telemetry.set_title('Closed-Loop Telemetry', color='white', fontsize=10, fontweight='bold')
        self.ax_telemetry.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
        self.ax_telemetry.set_ylabel('Surprise / Alpha', color='#cccccc', fontsize=8)
        self.ax_telemetry.grid(True, linestyle=':', alpha=0.2, color='#444444')
        
        self.ax_telemetry_right = self.ax_telemetry.twinx()
        self.ax_telemetry_right.set_facecolor('none')
        self.ax_telemetry_right.tick_params(colors='#cccccc', labelsize=8)
        for spine in self.ax_telemetry_right.spines.values():
            spine.set_edgecolor('#444444')
            
        self.line_repel, = self.ax_telemetry_right.plot([], [], '-', color='#ff33ff', linewidth=1.2, label='SOG Force (N)')
        self.line_emd, = self.ax_telemetry_right.plot([], [], '-', color='#39ff14', linewidth=1.2, label='EMD Force (N)')
        self.line_instar, = self.ax_telemetry_right.plot([], [], '-', color='#ffff33', linewidth=1.2, label='Instar Force (N)')
        
        self.ax_telemetry_right.set_ylabel('Force (Newtons)', color='#cccccc', fontsize=8)
        
        lines = [self.line_surprise, self.line_alpha, self.line_repel, self.line_emd, self.line_instar]
        labels = [l.get_label() for l in lines]
        self.ax_telemetry.legend(lines, labels, loc='upper left', facecolor='#222222', labelcolor='white', fontsize=7)
        self.telemetry_time_line = self.ax_telemetry.axvline(x=0, color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)
        
        plt.tight_layout()
        
    def onclick(self, event):
        if event.inaxes == self.ax_nav:
            if event.xdata is not None and event.ydata is not None:
                # Click coordinates in SLAM space -> convert to physical
                phys_x = (event.xdata - 1.0) / self.env._slam_scale
                phys_y = (event.ydata - 1.0) / self.env._slam_scale
                self.vis_target_xy = jnp.array([[phys_x, phys_y]])
                print(f"🎯 [TARGET CLICKED] SLAM: ({event.xdata:.2f}, {event.ydata:.2f}) -> Physical Target: ({phys_x:.2f}, {phys_y:.2f})")
                
                # Update target stars on map
                self.tgt_nav_marker.set_data([event.xdata], [event.ydata])
                self.tgt_map_marker.set_data([event.xdata], [event.ydata])
                self.fig.canvas.draw_idle()
                
    def on_close(self, event):
        print("--> Window closed by user.")
        self.is_running = False
        
    def on_key(self, event):
        if event.key in ('q', 'escape'):
            print(f"--> Exiting interactive mode via key '{event.key}'.")
            self.is_running = False
            plt.close(self.fig)
            
    def run(self):
        self.setup_plots()
        
        # Connect Matplotlib events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.ion()
        self.fig.show()
        
        # Simulation setup
        physics_dt = Config.DT * Config.SIM_SUBSTEPS # ~2.16 ms
        kin_acc = np.zeros(3, dtype=np.float32)
        kin_count = 0
        
        dr_x = self.start_slam_x
        dr_z = self.start_slam_z
        dr_th = self.start_slam_th
        raw_imu_x = self.start_slam_x
        raw_imu_z = self.start_slam_z
        raw_imu_th = self.start_slam_th
        slam_est_u = self.start_slam_x
        slam_est_v = self.start_slam_z
        slam_est_th = self.start_slam_th
        last_slam_est_u = self.start_slam_x
        last_slam_est_v = self.start_slam_z
        last_slam_est_th = self.start_slam_th
        
        last_sensor_heading = self.start_slam_th
        
        last_active_places = np.array([], dtype=np.int32)
        last_slam_surprise = 0.0
        last_smooth_omega = 0.0
        last_learn_gate = True
        slam_vis_csnn_jax = jnp.zeros((1, 256))
        slam_vis_stdp_jax = jnp.zeros((1, 256))
        
        slam_prev_int = np.zeros(N_PIXELS, dtype=np.float32)
        vis_prev_int = np.zeros(N_PIXELS, dtype=np.float32)
        
        _TOF_ANGLES = [-np.pi/4, 0.0, np.pi/4, np.pi]
        
        current_step_counter = 0
        vis_emd_intensities = jnp.zeros((1, Config.N_EMD_PIX))
        
        # Cumulative place cell spike raster records for plotting
        spike_t = []
        spike_idx = []
        
        print("\n🚀 [STARTING LIVE INTERACTIVE INFERENCE]")
        print("💡 Click anywhere in the Navigation Room panel to dynamically update the flight target.")
        print("💡 Press 'q' or close the window to exit and export the GIF history.")
        
        step_idx = 0
        t_start = time.time()
        
        while self.is_running:
            if self.max_steps is not None and step_idx >= self.max_steps:
                print(f"--> Reached step limit of {self.max_steps}. Exiting.")
                self.is_running = False
                plt.close(self.fig)
                break
            r_st_start = np.array(self.state[0][0])
            r_st = self.state[0]
            r_cpu = np.array(r_st[0])
            
            # Boundary/Collision check of current state before stepping
            is_oor = (abs(r_cpu[0]) > Config.ARENA_W) or (abs(r_cpu[1]) > Config.ARENA_W)
            slam_u = r_cpu[0] * self.env._slam_scale + 1.0
            slam_v = r_cpu[1] * self.env._slam_scale + 1.0
            _obs_np = np.array(self.env._obstacles) if self.env._obstacles is not None else np.zeros((0, 4))
            is_in_obs = False
            if len(_obs_np) > 0:
                is_in_obs = bool(np.any(
                    (_obs_np[:, 0] <= slam_u) & (_obs_np[:, 2] >= slam_u) &
                    (_obs_np[:, 1] <= slam_v) & (_obs_np[:, 3] >= slam_v)
                ))
            is_crashed = is_oor or is_in_obs
            
            if np.isnan(r_cpu).any() or is_crashed:
                if np.isnan(r_cpu).any():
                    print("!!! NaN detected in robot state. Resetting... !!!")
                else:
                    print(f"!!! Crash detected (OOR: {is_oor}, Obs: {is_in_obs}). Resetting... !!!")
                # Reset environment
                self.rng, key_reset = jax.random.split(self.rng)
                self.state = self.env.reset(key_reset, 1)
                r_state_override = self.state[0].at[0].set(Config.TARGET_STATE)
                self.state = (r_state_override,) + self.state[1:]
                # Reset SLAM
                self.vis_slam.reset_pose_only(1)
                self.vis_slam.initialize_pose(
                    jnp.array([[self.start_slam_x, self.start_slam_z]]),
                    jnp.array([self.start_slam_th]),
                )
                dr_x, dr_z, dr_th = self.start_slam_x, self.start_slam_z, self.start_slam_th
                raw_imu_x, raw_imu_z, raw_imu_th = self.start_slam_x, self.start_slam_z, self.start_slam_th
                slam_est_u, slam_est_v, slam_est_th = self.start_slam_x, self.start_slam_z, self.start_slam_th
                last_slam_est_u, last_slam_est_v, last_slam_est_th = self.start_slam_x, self.start_slam_z, self.start_slam_th
                last_sensor_heading = self.start_slam_th
                last_slam_surprise = 0.0
                self.env._prev_robot_state = None
                r_st_start = np.array(self.state[0][0])
                r_st = self.state[0]
                r_cpu = np.array(r_st[0])
            
            slam_pose_jax = jnp.array([[slam_est_u, slam_est_v, slam_est_th]])
            vis_step_counter = current_step_counter + Config.WARMUP_STEPS + 5
            
            # Step physics surrogate and controller
            self.state, f_nodal, w_pose, h_marker, alpha_floored_jax, f_repel_scaled_jax, flow_corr_jax, vis_emd_intensities, f_net_slam_jax, f_brain_slam_jax, f_emd_slam_jax, f_instar_slam_jax = vis_step_fn(
                self.env, self.state, self.params_single, vis_step_counter, jnp.array([last_slam_surprise]), self.hover_params_single, slam_pose_jax, slam_vis_csnn_jax, slam_vis_stdp_jax, self._sog_state.v_mem, Config.K_REPEL, Config.K_FLOW, self.vis_target_xy, vis_emd_intensities
            )
            current_step_counter += 1
            
            r_state_np = np.array(self.state[0][0])
            t = current_step_counter * physics_dt
            
            # Boundary check
            is_oor = (abs(r_state_np[0]) > Config.ARENA_W) or (abs(r_state_np[1]) > Config.ARENA_W)
            slam_u = r_state_np[0] * self.env._slam_scale + 1.0
            slam_v = r_state_np[1] * self.env._slam_scale + 1.0
            
            # Dead reckoning
            sensor_heading = float(r_state_np[2]) - 1.0
            # Position: use raw state velocities (models IMU/optic-flow velocity sensor)
            dr_vx = float(r_state_np[4]) * self.env._slam_scale
            dr_vz = float(r_state_np[5]) * self.env._slam_scale
            # Heading: use AHRS-style heading differences (filters wingbeat oscillations)
            dr_vth = (sensor_heading - last_sensor_heading + np.pi) % (2 * np.pi) - np.pi
            dr_vth = dr_vth / physics_dt
            
            last_sensor_heading = sensor_heading
            
            raw_imu_x += dr_vx * physics_dt
            raw_imu_z += dr_vz * physics_dt
            raw_imu_th += dr_vth * physics_dt
            raw_imu_th = (raw_imu_th + np.pi) % (2 * np.pi) - np.pi
            
            # Correction filter
            K_CORR = Config.K_CORR
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
            
            # Capture starting state for 10-step blocks
            if step_idx % 10 == 0:
                r_st_start_10 = np.array(r_st_start)
                
            _tof_jax = compute_tof_distance(
                jnp.array([slam_u, slam_v]), sensor_heading, self.env._segments, include_back=True
            )
            
            # Accumulate kinematics at high frequency (460Hz) to prevent aliasing
            cos_sh = np.cos(sensor_heading)
            sin_sh = np.sin(sensor_heading)
            vx_sensor = (r_state_np[4] * cos_sh + r_state_np[5] * sin_sh) * self.env._slam_scale
            vz_sensor = (-r_state_np[4] * sin_sh + r_state_np[5] * cos_sh) * self.env._slam_scale
            w_theta = r_state_np[6]
            kin_acc += np.array([vx_sensor, vz_sensor, w_theta], dtype=np.float32)
            kin_count += 1

            # CANN SLAM Update (every 10 control steps, 50Hz)
            if (step_idx + 1) % 10 == 0:
                elapsed_time = 10 * physics_dt
                self.env._prev_robot_state = r_st_start_10
                
                avg_w_theta = kin_acc[2] / kin_count if kin_count > 0 else 0.0
                ev_jax, kin_jax, tof_jax, acc_jax, slam_prev_int = self.env.compute_slam_sensors(
                    r_state_np, slam_prev_int, dt=elapsed_time, override_w_theta=avg_w_theta
                )
                
                try:
                    pose_est, _, _, _, _, debug_gates = self.vis_slam.forward_step(
                        ev_jax, kin_jax, tof_jax,
                        acc_t=acc_jax,
                        inject_drift=False, autopilot_on=True,
                        dt=elapsed_time
                    )
                    raw_match = float(debug_gates['Raw_Match'][0])
                    conc_place = float(debug_gates['Conc_Place'][0])
                    composite_match = raw_match
                    last_slam_surprise = float(1.0 - np.exp(-5.0 * (1.0 - composite_match)))
                    
                    last_smooth_omega = float(debug_gates['Smooth_Omega'][0])
                    last_learn_gate = bool(debug_gates['Learn_Gate'])
                    
                    last_slam_est_u = float(pose_est[0, 0])
                    last_slam_est_v = float(pose_est[0, 1])
                    last_slam_est_th = float(pose_est[0, 2])
                    
                    slam_vis_csnn_jax = jnp.array(debug_gates['Debug_Input_CSNN'])
                    slam_vis_stdp_jax = jnp.array(debug_gates['Debug_Input_STDP'])
                    
                    I_place_np = np.array(debug_gates['Debug_I_Place'][0])
                    max_val = np.max(I_place_np)
                    last_active_places = np.where(I_place_np > 0.1 * max_val)[0] if max_val > 1e-5 else np.array([], dtype=np.int32)
                    
                    # Accumulate spike points
                    for neuron_id in last_active_places:
                        spike_t.append(t)
                        spike_idx.append(neuron_id)
                        
                except Exception as e:
                    print(f"!!! CANN SLAM error: {e}")
                
                # Reset accumulators
                kin_acc = np.zeros(3, dtype=np.float32)
                kin_count = 0
                    
            # Compute 1D event camera stream for visualization
            _vis_slam_pos = jnp.array([slam_u, slam_v])
            _vis_int, _, _, _ = compute_pixel_readings(
                _vis_slam_pos, sensor_heading, self.env._segments,
                obstacles=self.env._obstacles, tex_tensor=self.env._tex_tensor
            )
            _vis_int_np = np.array(_vis_int)
            _vis_delta = _vis_int_np - vis_prev_int
            _vis_events = np.where(_vis_delta > THRESHOLD, 1.0,
                         np.where(_vis_delta < -THRESHOLD, -1.0, 0.0)).astype(np.float32)
            vis_prev_int = _vis_int_np
            
            # SOG Update
            tof_d = np.array(_tof_jax)
            _hit_idx, _free_idx = _get_ray_indices(
                slam_est_u, slam_est_v, slam_est_th,
                tof_d, _TOF_ANGLES,
                res=self._sog.res, grid_size=self._sog.grid_w, offset_m=self._sog.offset_m
            )
            self._sog_state = self._sog.update(self._sog_state, jnp.array(_hit_idx), jnp.array(_free_idx))
            
            # Record trajectory
            self.sim_data['states'].append(r_state_np)
            self.sim_data['wing_pose'].append(np.array(w_pose[0]))
            self.sim_data['nodal_forces'].append(np.array(f_nodal[0]))
            self.sim_data['le_marker'].append(np.array(self.state[1].marker_le[0]))
            self.sim_data['hinge_marker'].append(np.array(h_marker[0]))
            self.sim_data['t'].append(t)
            self.sim_data['slam_pos'].append((slam_u, slam_v))
            self.sim_data['slam_est'].append((slam_est_u, slam_est_v, slam_est_th))
            self.sim_data['imu_dr'].append((raw_imu_x, raw_imu_z, raw_imu_th))
            self.sim_data['surprise'].append(last_slam_surprise)
            self.sim_data['smooth_omega'].append(last_smooth_omega)
            self.sim_data['learn_gate'].append(last_learn_gate)
            self.sim_data['tof'].append(tof_d)
            self.sim_data['heading'].append(sensor_heading)
            self.sim_data['events'].append(_vis_events)
            self.sim_data['active_places'].append(last_active_places)
            self.sim_data['alpha'].append(float(alpha_floored_jax.squeeze()))
            self.sim_data['f_repel'].append(float(jnp.linalg.norm(f_repel_scaled_jax.squeeze())))
            self.sim_data['flow_corr'].append(float(flow_corr_jax.squeeze()))
            self.sim_data['f_repel_vec'].append(np.array(f_repel_scaled_jax.squeeze()))
            self.sim_data['f_brain_vec'].append(np.array(f_brain_slam_jax.squeeze()))
            self.sim_data['f_net_vec'].append(np.array(f_net_slam_jax.squeeze()))
            self.sim_data['f_emd_vec'].append(np.array(f_emd_slam_jax.squeeze()))
            self.sim_data['f_instar_vec'].append(np.array(f_instar_slam_jax.squeeze()))
            self.sim_data['sog_states'].append(np.array(self._sog_state.v_mem))
            self.sim_data['target_xy'].append(np.array(self.vis_target_xy[0]))
            
            # --- Live Render update ---
            if step_idx % self.draw_skip == 0:
                frame_idx = len(self.sim_data['states']) - 1
                
                # 1. Navigation Room
                self.traj_line.set_data(
                    [p[0] for p in self.sim_data['slam_pos']],
                    [p[1] for p in self.sim_data['slam_pos']]
                )
                self.hornet_dot.set_data([slam_u], [slam_v])
                
                self.slam_est_line.set_data(
                    [p[0] for p in self.sim_data['slam_est']],
                    [p[1] for p in self.sim_data['slam_est']]
                )
                self.slam_est_dot.set_data([slam_est_u], [slam_est_v])
                
                self.imu_dr_line.set_data(
                    [p[0] for p in self.sim_data['imu_dr']],
                    [p[1] for p in self.sim_data['imu_dr']]
                )
                self.imu_dr_dot.set_data([raw_imu_x], [raw_imu_z])
                
                self.heading_arr.set_offsets([[slam_u, slam_v]])
                self.heading_arr.set_UVC([0.2 * np.cos(sensor_heading)], [0.2 * np.sin(sensor_heading)])
                
                self.slam_heading_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.slam_heading_arr.set_UVC([0.2 * np.cos(slam_est_th)], [0.2 * np.sin(slam_est_th)])
                
                vx_phys, vz_phys = r_state_np[4], r_state_np[5]
                self.vel_arr.set_offsets([[slam_u, slam_v]])
                self.vel_arr.set_UVC([0.2 * vx_phys * self.env._slam_scale], [0.2 * vz_phys * self.env._slam_scale])
                
                tgt_u = float(self.vis_target_xy[0, 0]) * self.env._slam_scale + 1.0
                tgt_v = float(self.vis_target_xy[0, 1]) * self.env._slam_scale + 1.0
                
                dx_believed = tgt_u - slam_est_u
                dy_believed = tgt_v - slam_est_v
                dist_believed = np.sqrt(dx_believed**2 + dy_believed**2) + 1e-8
                ux_believed = dx_believed / dist_believed
                uy_believed = dy_believed / dist_believed
                
                self.target_dir_arr.set_offsets([[slam_u, slam_v]])
                self.target_dir_arr.set_UVC([0.2 * ux_believed], [0.2 * uy_believed])
                
                # ToF beams (4 beams: -45, 0, 45, 180 deg)
                _tof_offsets = [-np.pi/4, 0.0, np.pi/4, np.pi]
                for bi, ((bl, bm), offset) in enumerate(zip(self.tof_beam_artists, _tof_offsets)):
                    ang = sensor_heading + offset
                    hu = slam_u + tof_d[bi] * np.cos(ang)
                    hv = slam_v + tof_d[bi] * np.sin(ang)
                    bl.set_data([slam_u, hu], [slam_v, hv])
                    bm.set_data([hu], [hv])
                    
                # Camera cone
                fov_r = 1.2
                self.fov_left_line.set_data(
                    [slam_u, slam_u + fov_r * np.cos(sensor_heading - np.pi/4)],
                    [slam_v, slam_v + fov_r * np.sin(sensor_heading - np.pi/4)]
                )
                self.fov_right_line.set_data(
                    [slam_u, slam_u + fov_r * np.cos(sensor_heading + np.pi/4)],
                    [slam_v, slam_v + fov_r * np.sin(sensor_heading + np.pi/4)]
                )
                
                # Highlight obstacles
                for obs_bbox, op in self.obs_patch_list:
                    x0, y0, x1, y1 = obs_bbox
                    inside = (x0 <= slam_u <= x1) and (y0 <= slam_v <= y1)
                    op.set_facecolor('#cc1111' if inside else '#2a2a4c')
                    op.set_edgecolor('#ff3333' if inside else '#6666bb')
                    
                # Event camera strip
                self.vis_strip.set_data(_vis_events[None, :])
                
                # Status text
                alpha_val = float(alpha_floored_jax.squeeze())
                self.nav_time.set_text(
                    f'θ={sensor_heading:.2f}r | ToF [{tof_d[0]:.1f}│{tof_d[1]:.1f}│{tof_d[2]:.1f}│{tof_d[3]:.1f}]m | Surprise={last_slam_surprise:.2f} | α={alpha_val:.2f}'
                )
                
                # 2. SOG map
                self.occ_img.set_data(self._sog_state.v_mem.T)
                self.occ_img.set_clim(vmin=-0.2, vmax=1.0)
                self.map_robot_dot.set_data([slam_est_u], [slam_est_v])
                
                self.map_trail_line.set_data(
                    [p[0] for p in self.sim_data['slam_est']],
                    [p[1] for p in self.sim_data['slam_est']]
                )
                
                self.map_target_dir_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_target_dir_arr.set_UVC([0.2 * ux_believed], [0.2 * uy_believed])
                
                f_scale = 6.25
                fr_x, fr_y = np.array(f_repel_scaled_jax.squeeze())
                self.map_f_repel_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_f_repel_arr.set_UVC([f_scale * fr_x], [f_scale * fr_y])
                
                fb_x, fb_y = np.array(f_brain_slam_jax.squeeze())
                self.map_f_brain_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_f_brain_arr.set_UVC([f_scale * fb_x], [f_scale * fb_y])
                
                fe_x, fe_y = np.array(f_emd_slam_jax.squeeze())
                self.map_f_emd_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_f_emd_arr.set_UVC([f_scale * fe_x], [f_scale * fe_y])
                
                fi_x, fi_y = np.array(f_instar_slam_jax.squeeze())
                self.map_f_instar_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_f_instar_arr.set_UVC([f_scale * fi_x], [f_scale * fi_y])
                
                fn_x, fn_y = np.array(f_net_slam_jax.squeeze())
                self.map_f_net_arr.set_offsets([[slam_est_u, slam_est_v]])
                self.map_f_net_arr.set_UVC([f_scale * fn_x], [f_scale * fn_y])
                
                _tof_offsets = [-np.pi/4, 0.0, np.pi/4, np.pi]
                for bi, ((bl, bm), offset) in enumerate(zip(self.map_tof_beam_artists, _tof_offsets)):
                    ang = slam_est_th + offset
                    hu = slam_est_u + tof_d[bi] * np.cos(ang)
                    hv = slam_est_v + tof_d[bi] * np.sin(ang)
                    bl.set_data([slam_est_u, hu], [slam_est_v, hv])
                    bm.set_data([hu], [hv])
                    
                self.map_fov_left_line.set_data(
                    [slam_est_u, slam_est_u + fov_r * np.cos(slam_est_th - np.pi/4)],
                    [slam_est_v, slam_est_v + fov_r * np.sin(slam_est_th - np.pi/4)]
                )
                self.map_fov_right_line.set_data(
                    [slam_est_u, slam_est_u + fov_r * np.cos(slam_est_th + np.pi/4)],
                    [slam_est_v, slam_est_v + fov_r * np.sin(slam_est_th + np.pi/4)]
                )
                
                # 3. Wing Mechanics
                rx, rz = r_state_np[0], r_state_np[1]
                r_th, r_phi = r_state_np[2], r_state_np[3]
                self.ax_wing.set_xlim(rx - 0.25, rx + 0.25)
                self.ax_wing.set_ylim(rz - 0.25, rz + 0.25)
                
                for obs_phys, op_wing in self.obs_patch_wing_list:
                    px0, py0, px1, py1 = obs_phys
                    inside = (px0 <= rx <= px1) and (py0 <= rz <= py1)
                    op_wing.set_facecolor('#cc1111' if inside else '#2a2a4c')
                    op_wing.set_edgecolor('#ff3333' if inside else '#6666bb')
                    
                active_props_batch = self.state[3]
                real_props = jax.tree.map(lambda x: x[0], active_props_batch)
                d1 = real_props.d1
                d2 = real_props.d2
                
                self.patch_thorax.set_center((rx, rz))
                self.patch_thorax.set_angle(np.degrees(r_th))
                self.patch_head.set_center((rx + d1 * np.cos(r_th), rz + d1 * np.sin(r_th)))
                
                joint_x = rx - d1 * np.cos(r_th)
                joint_z = rz - d1 * np.sin(r_th)
                abd_ang = r_th + r_phi
                self.patch_abd.set_center((joint_x - d2 * np.cos(abd_ang), joint_z - d2 * np.sin(abd_ang)))
                self.patch_abd.set_angle(np.degrees(abd_ang))
                
                frame_idx = len(self.sim_data['states']) - 1
                prev1_idx = max(0, frame_idx - 1)
                prev2_idx = max(0, frame_idx - 2)
                
                # 1. Retrieve current and previous states
                r_state_curr = self.sim_data['states'][frame_idx]
                r_state_prev1 = self.sim_data['states'][prev1_idx]
                r_state_prev2 = self.sim_data['states'][prev2_idx]
                
                rx_curr, rz_curr, r_th_curr = r_state_curr[0], r_state_curr[1], r_state_curr[2]
                
                # Project prev2 wing pose to current coordinates
                rx_p2, rz_p2, r_th_p2 = r_state_prev2[0], r_state_prev2[1], r_state_prev2[2]
                wx_p2, wz_p2, wang_p2 = self.sim_data['wing_pose'][prev2_idx]
                
                dx2 = wx_p2 - rx_p2
                dz2 = wz_p2 - rz_p2
                cos2, sin2 = np.cos(r_th_p2), np.sin(r_th_p2)
                dx_body2 = dx2 * cos2 + dz2 * sin2
                dz_body2 = -dx2 * sin2 + dz2 * cos2
                wang_body2 = wang_p2 - r_th_p2
                
                cos_c, sin_c = np.cos(r_th_curr), np.sin(r_th_curr)
                dx_proj2 = dx_body2 * cos_c - dz_body2 * sin_c
                dz_proj2 = dx_body2 * sin_c + dz_body2 * cos_c
                wx_proj_prev2 = rx_curr + dx_proj2
                wz_proj_prev2 = rz_curr + dz_proj2
                wang_proj_prev2 = r_th_curr + wang_body2
                
                # Project prev1 wing pose to current coordinates
                rx_p1, rz_p1, r_th_p1 = r_state_prev1[0], r_state_prev1[1], r_state_prev1[2]
                wx_p1, wz_p1, wang_p1 = self.sim_data['wing_pose'][prev1_idx]
                
                dx1 = wx_p1 - rx_p1
                dz1 = wz_p1 - rz_p1
                cos1, sin1 = np.cos(r_th_p1), np.sin(r_th_p1)
                dx_body1 = dx1 * cos1 + dz1 * sin1
                dz_body1 = -dx1 * sin1 + dz1 * cos1
                wang_body1 = wang_p1 - r_th_p1
                
                dx_proj1 = dx_body1 * cos_c - dz_body1 * sin_c
                dz_proj1 = dx_body1 * sin_c + dz_body1 * cos_c
                wx_proj_prev1 = rx_curr + dx_proj1
                wz_proj_prev1 = rz_curr + dz_proj1
                wang_proj_prev1 = r_th_curr + wang_body1
                
                # Current wing pose
                wx_proj_curr, wz_proj_curr, wang_proj_curr = self.sim_data['wing_pose'][frame_idx]
                
                # 4. Interpolate 9 poses over 0.5 cycles (last 2 control steps)
                num_w = len(self.wing_lines)
                for k in range(num_w):
                    tau = k / float(num_w - 1) if num_w > 1 else 1.0
                    if tau <= 0.5:
                        alpha = 2.0 * tau
                        wx_interp = (1.0 - alpha) * wx_proj_prev2 + alpha * wx_proj_prev1
                        wz_interp = (1.0 - alpha) * wz_proj_prev2 + alpha * wz_proj_prev1
                        
                        d_theta = wang_proj_prev1 - wang_proj_prev2
                        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
                        wang_interp = wang_proj_prev2 + alpha * d_theta
                    else:
                        alpha = 2.0 * (tau - 0.5)
                        wx_interp = (1.0 - alpha) * wx_proj_prev1 + alpha * wx_proj_curr
                        wz_interp = (1.0 - alpha) * wz_proj_prev1 + alpha * wz_proj_curr
                        
                        d_theta = wang_proj_curr - wang_proj_prev1
                        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
                        wang_interp = wang_proj_prev1 + alpha * d_theta
                    
                    wing_len = self.env.phys.fluid.WING_LEN
                    N_pts    = self.env.phys.fluid.N_PTS
                    x_local  = np.linspace(wing_len/2, -wing_len/2, N_pts)
                    c_w, s_w = np.cos(wang_interp), np.sin(wang_interp)
                    wing_x   = wx_interp + x_local * c_w
                    wing_z   = wz_interp + x_local * s_w
                    self.wing_lines[k].set_data(wing_x, wing_z)
                    
                    if k == num_w - 1:
                        self.patch_le.set_center((wing_x[0], wing_z[0]))
                        self.patch_hinge.set_center((h_marker[0][0], h_marker[0][1]))
                        
                        pts = np.stack([wing_x, wing_z], axis=1)
                        self.quiver.set_offsets(pts)
                        self.quiver.set_UVC(f_nodal[0][:, 0], f_nodal[0][:, 1])
                        
                self.time_text.set_text(f'T={t:.4f}s | Z={rz:.3f}m')
                
                # 4. SNN Raster
                self.snn_spike_line.set_data(spike_t, spike_idx)
                self.ax_snn.set_xlim(0, max(5.0, t + 0.5))
                self.snn_time_line.set_xdata([t])
                
                # 5. Telemetry
                time_series = self.sim_data['t']
                self.line_surprise.set_data(time_series, self.sim_data['surprise'])
                self.line_alpha.set_data(time_series, self.sim_data['alpha'])
                
                # Plot all forces in physical Newtons
                f_repel_np = np.array(self.sim_data['f_repel'])
                self.line_repel.set_data(time_series, f_repel_np)
                
                f_emd_np = np.array([np.linalg.norm(v) for v in self.sim_data['f_emd_vec']])
                self.line_emd.set_data(time_series, f_emd_np)
                
                f_instar_np = np.array([np.linalg.norm(v) for v in self.sim_data['f_instar_vec']])
                self.line_instar.set_data(time_series, f_instar_np)
                
                max_val = 0.30  # stable maximum for force y-limit
                
                self.ax_telemetry.set_xlim(0, max(5.0, t + 0.5))
                self.ax_telemetry_right.set_ylim(-0.01, max_val * 1.1)
                self.telemetry_time_line.set_xdata([t])
                
                fig_title = f"Dashboard - Running: {(time.time() - t_start):.1f}s | Simulator Time: {t:.2f}s | Steps: {step_idx}"
                self.fig.suptitle(fig_title, color='white', fontsize=12, fontweight='bold')
                
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
                
            step_idx += 1
            
        print(f"\n✅ [SIMULATION ENDED] Ran {step_idx} control steps ({t:.2f} seconds of simulator time).")
        self.save_gif()
        
    def save_gif(self):
        n_frames = len(self.sim_data['states'])
        if n_frames == 0:
            print("--> Inference ended before any frames were recorded. GIF not saved.")
            return
            
        MAX_GIF_FRAMES = 600
        if n_frames > MAX_GIF_FRAMES:
            print(f"--> Truncating recorded data from {n_frames} frames to the last {MAX_GIF_FRAMES} frames.")
            for key in self.sim_data:
                if isinstance(self.sim_data[key], list):
                    self.sim_data[key] = self.sim_data[key][-MAX_GIF_FRAMES:]
            n_frames = MAX_GIF_FRAMES

        print(f"--> Exporting trajectory of {n_frames} frames to GIF: {self.save_gif_path}")
        print("    (This may take a moment to compile, please wait...)")
        
        # Turn off interactive plotting
        plt.ioff()
        
        # Re-initialize dashboard figure for animation rendering
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#0d0d0d')
        gs = fig.add_gridspec(2, 3)
        
        ax_nav = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[1, 0])
        ax_wing = fig.add_subplot(gs[0, 1])
        ax_snn = fig.add_subplot(gs[1, 1])
        ax_telemetry = fig.add_subplot(gs[0, 2])
        ax_vis = fig.add_subplot(gs[1, 2])
        
        for ax in (ax_nav, ax_map, ax_wing, ax_snn, ax_telemetry, ax_vis):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='#aaaaaa', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444444')
                
        # Room boundaries and static elements
        ax_nav.set_xlim(0, 2)
        ax_nav.set_ylim(0, 2)
        ax_nav.set_aspect('equal')
        ax_nav.set_title('Navigation Room (SLAM Space)', color='white', fontsize=10, fontweight='bold')
        ax_nav.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
        ax_nav.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)
        
        room_rect = plt.Rectangle((0, 0), 2, 2, linewidth=2, edgecolor='#00ffcc', facecolor='none')
        ax_nav.add_patch(room_rect)
        
        obstacles_np = np.array(self.env._obstacles) if self.env._obstacles is not None else np.zeros((0, 4))
        obs_patch_list = []
        for obs in obstacles_np:
            x0, y0, x1, y1 = obs
            p = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor='#2a2a4c', edgecolor='#6666bb', linewidth=1, zorder=4)
            ax_nav.add_patch(p)
            obs_patch_list.append((obs, p))
            
        # Target mark
        tgt_nav_marker, = ax_nav.plot([], [], marker='*', markersize=14, color='#ffdd00',
                                            markeredgecolor='#ff8800', zorder=20, label='Target')
        
        # Trajectories and lines
        traj_line, = ax_nav.plot([], [], '-', color='#00ff88', linewidth=1.0, alpha=0.6, zorder=5, label='Path (GT)')
        hornet_dot, = ax_nav.plot([], [], 'o', color='#ff4444', markersize=7, zorder=15, label='Hornet (GT)')
        slam_est_line, = ax_nav.plot([], [], ':', color='#ffa500', linewidth=1.2, alpha=0.8, zorder=6, label='Path (SLAM)')
        slam_est_dot, = ax_nav.plot([], [], 's', color='#ffa500', markersize=5, zorder=14, label='SLAM Pose')
        imu_dr_line, = ax_nav.plot([], [], '--', color='#00ffff', linewidth=0.8, alpha=0.5, zorder=4, label='Path (IMU DR)')
        imu_dr_dot, = ax_nav.plot([], [], '^', color='#00ffff', markersize=5, zorder=13, label='IMU DR Pose')
        
        heading_arr = ax_nav.quiver([0], [0], [0], [0], color='#ff8888', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Sensor Heading')
        slam_heading_arr = ax_nav.quiver([0], [0], [0], [0], color='#ffa500', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=15, label='SLAM Heading')
        vel_arr = ax_nav.quiver([0], [0], [0], [0], color='#ff33ff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Velocity')
        target_dir_arr = ax_nav.quiver([0], [0], [0], [0], color='#ffcc00', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Believed Target Dir')
        
        # ToF Beams
        tof_beam_artists = []
        _beam_colours = ['#00aaff', '#ffff44', '#00aaff', '#ff5555']
        for _bc in _beam_colours:
            _bl, = ax_nav.plot([], [], '-',  color=_bc, linewidth=1.5, alpha=0.85, zorder=12)
            _bm, = ax_nav.plot([], [], 'D',  color=_bc, markersize=4,  alpha=0.95, zorder=13)
            tof_beam_artists.append((_bl, _bm))
            
        fov_left_line,  = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
        fov_right_line, = ax_nav.plot([], [], '--', color='#ffff88', linewidth=0.8, alpha=0.4, zorder=3)
        
        nav_time = ax_nav.text(0.02, 0.97, '', transform=ax_nav.transAxes,
                               color='#cccccc', fontsize=8, va='top', family='monospace')
        ax_nav.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)
        
        # 1D Event Strip
        ax_vis.set_xticks([])
        ax_vis.set_yticks([])
        ax_vis.set_title('1D Event Camera Stream (Green=ON, Red=OFF)', color='white', fontsize=10, fontweight='bold')
        colors_ev = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.8, 0.1)]
        cm_ev = LinearSegmentedColormap.from_list('events_cmap', colors_ev, N=3)
        vis_strip = ax_vis.imshow(np.zeros((1, N_PIXELS)), cmap=cm_ev, vmin=-1.0, vmax=1.0, aspect='auto')
        
        # Map
        ax_map.set_aspect('equal')
        ax_map.set_title('Spiking Occupancy Grid (Robot Memory)', color='white', fontsize=10, fontweight='bold')
        ax_map.set_xlabel('X (m)', color='#aaaaaa', fontsize=8)
        ax_map.set_ylabel('Y (m)', color='#aaaaaa', fontsize=8)
        
        SOG_EXTENT = [0, 2, 0, 2]
        occ_img = ax_map.imshow(
            np.zeros((self._sog.grid_w, self._sog.grid_w), dtype=np.float32), 
            origin='lower', extent=SOG_EXTENT,
            cmap='magma', vmin=-0.2, vmax=1.0, aspect='equal', zorder=1
        )
        map_robot_dot, = ax_map.plot([], [], 'o', color='#00ffcc', markersize=5, zorder=10, label='Robot')
        map_trail_line, = ax_map.plot([], [], '-', color='#00ffcc', linewidth=0.8, alpha=0.4, zorder=5, label='Trail')
        tgt_map_marker, = ax_map.plot([], [], marker='*', markersize=14, color='#ffdd00',
                                            markeredgecolor='#ff8800', zorder=20, label='Target')
        
        map_target_dir_arr = ax_map.quiver([0], [0], [0], [0], color='#ffcc00', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=16, label='Believed Target Dir')
        map_f_repel_arr = ax_map.quiver([0], [0], [0], [0], color='#ff33ff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='SOG Repulsion Force')
        map_f_brain_arr = ax_map.quiver([0], [0], [0], [0], color='#ff8800', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Brain Goal Force')
        map_f_emd_arr = ax_map.quiver([0], [0], [0], [0], color='#39ff14', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='EMD Reflex Force')
        map_f_instar_arr = ax_map.quiver([0], [0], [0], [0], color='#ff007f', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=17, label='Instar Memory Force')
        map_f_net_arr = ax_map.quiver([0], [0], [0], [0], color='#00ffff', scale=1.0, scale_units='xy', width=0.006, headwidth=4, headlength=5, zorder=18, label='Net Control Force')
        
        map_tof_beam_artists = []
        for _bc in _beam_colours:
            _bl, = ax_map.plot([], [], '-',  color=_bc, linewidth=1.2, alpha=0.6, zorder=12)
            _bm, = ax_map.plot([], [], 'D',  color=_bc, markersize=3,  alpha=0.7, zorder=13)
            map_tof_beam_artists.append((_bl, _bm))
            
        map_fov_left_line,  = ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
        map_fov_right_line, = ax_map.plot([], [], '--', color='#ffff88', linewidth=0.6, alpha=0.3, zorder=3)
        ax_map.legend(loc='lower right', facecolor='#222222', labelcolor='white', fontsize=8)
        
        # Wing
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
        
        # --- WINGS (Ghost/Shadow trail to solve aliasing: 9 poses spanning 0.5 cycles) ---
        wing_lines = []
        shadow_alphas = np.linspace(0.05, 0.4, 8)
        all_alphas = np.concatenate([shadow_alphas, [1.0]])
        for a in all_alphas:
            lw = 1.5 if a == 1.0 else 1.0
            col = '#ffffff' if a == 1.0 else '#cccccc'
            wl, = ax_wing.plot([], [], col, linestyle='-', linewidth=lw, alpha=a, zorder=11 if a == 1.0 else 10)
            wing_lines.append(wl)
            
        patch_le    = patches.Circle((0,0), radius=0.001, color='#ff4444', zorder=15)
        patch_hinge = patches.Circle((0,0), radius=0.001, color='#ff8800', zorder=15)
        ax_wing.add_patch(patch_le)
        ax_wing.add_patch(patch_hinge)
        
        dummy = np.zeros(20)
        quiver = ax_wing.quiver(dummy, dummy, dummy, dummy, color='#ff6666',
                                 scale=3.0, scale_units='xy', zorder=20, width=0.002)
        time_text = ax_wing.text(0.02, 0.95, '', transform=ax_wing.transAxes, color='#cccccc', fontsize=8)
        
        # SNN Raster
        snn_spike_line, = ax_snn.plot([], [], 'o', color='#00ffff', markersize=2, alpha=0.4, linestyle='None')
        ax_snn.set_xlim(self.sim_data['t'][0], self.sim_data['t'][-1])
        ax_snn.set_ylim(-5, 260)
        ax_snn.set_title('Place Cell Spike Raster', color='white', fontsize=10, fontweight='bold')
        ax_snn.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
        ax_snn.set_ylabel('Neuron ID (0-255)', color='#aaaaaa', fontsize=8)
        ax_snn.grid(True, linestyle=':', alpha=0.2, color='#444444')
        snn_time_line = ax_snn.axvline(x=self.sim_data['t'][0], color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)
        
        # Telemetry
        line_surprise, = ax_telemetry.plot([], [], '-', color='#ffa500', linewidth=1.2, label='SLAM Surprise')
        line_alpha, = ax_telemetry.plot([], [], '-', color='#00ffff', linewidth=1.2, label='DNAG Alpha (Gate)')
        ax_telemetry.set_xlim(self.sim_data['t'][0], self.sim_data['t'][-1])
        ax_telemetry.set_ylim(-0.05, 1.1)
        ax_telemetry.set_title('Closed-Loop Telemetry', color='white', fontsize=10, fontweight='bold')
        ax_telemetry.set_xlabel('Time (s)', color='#aaaaaa', fontsize=8)
        ax_telemetry.set_ylabel('Surprise / Alpha', color='#cccccc', fontsize=8)
        ax_telemetry.grid(True, linestyle=':', alpha=0.2, color='#444444')
        
        ax_telemetry_right = ax_telemetry.twinx()
        ax_telemetry_right.set_facecolor('none')
        ax_telemetry_right.tick_params(colors='#cccccc', labelsize=8)
        for spine in ax_telemetry_right.spines.values():
            spine.set_edgecolor('#444444')
            
        line_repel, = ax_telemetry_right.plot([], [], '-', color='#ff33ff', linewidth=1.2, label='SOG Force (N)')
        line_emd, = ax_telemetry_right.plot([], [], '-', color='#39ff14', linewidth=1.2, label='EMD Force (N)')
        line_instar, = ax_telemetry_right.plot([], [], '-', color='#ffff33', linewidth=1.2, label='Instar Force (N)')
        
        ax_telemetry_right.set_ylabel('Force (Newtons)', color='#cccccc', fontsize=8)
        
        lines = [line_surprise, line_alpha, line_repel, line_emd, line_instar]
        labels = [l.get_label() for l in lines]
        ax_telemetry.legend(lines, labels, loc='upper left', facecolor='#222222', labelcolor='white', fontsize=7)
        telemetry_time_line = ax_telemetry.axvline(x=self.sim_data['t'][0], color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.8)
        
        plt.tight_layout()
        
        # Build compile lists of place cell spikes
        anim_spike_t = []
        anim_spike_idx = []
        
        def update_frame(frame):
            r_state = self.sim_data['states'][frame]
            w_pose = self.sim_data['wing_pose'][frame]
            f_nodal = self.sim_data['nodal_forces'][frame]
            hinge_p = self.sim_data['hinge_marker'][frame]
            t = self.sim_data['t'][frame]
            
            slam_pos = self.sim_data['slam_pos'][:frame+1]
            slam_u, slam_v = self.sim_data['slam_pos'][frame]
            slam_est = self.sim_data['slam_est'][:frame+1]
            slam_est_u, slam_est_v, slam_est_th = self.sim_data['slam_est'][frame]
            imu_dr = self.sim_data['imu_dr'][:frame+1]
            raw_imu_x, raw_imu_z, _ = self.sim_data['imu_dr'][frame]
            
            tof_d = self.sim_data['tof'][frame]
            sensor_heading = self.sim_data['heading'][frame]
            
            # Update target star based on recorded targets at this frame
            tgt_phys_f = self.sim_data['target_xy'][frame]
            tgt_u_f = tgt_phys_f[0] * self.env._slam_scale + 1.0
            tgt_v_f = tgt_phys_f[1] * self.env._slam_scale + 1.0
            tgt_nav_marker.set_data([tgt_u_f], [tgt_v_f])
            tgt_map_marker.set_data([tgt_u_f], [tgt_v_f])
            
            # Nav room lines
            traj_line.set_data([p[0] for p in slam_pos], [p[1] for p in slam_pos])
            hornet_dot.set_data([slam_u], [slam_v])
            slam_est_line.set_data([p[0] for p in slam_est], [p[1] for p in slam_est])
            slam_est_dot.set_data([slam_est_u], [slam_est_v])
            imu_dr_line.set_data([p[0] for p in imu_dr], [p[1] for p in imu_dr])
            imu_dr_dot.set_data([raw_imu_x], [raw_imu_z])
            
            heading_arr.set_offsets([[slam_u, slam_v]])
            heading_arr.set_UVC([0.2 * np.cos(sensor_heading)], [0.2 * np.sin(sensor_heading)])
            
            slam_heading_arr.set_offsets([[slam_est_u, slam_est_v]])
            slam_heading_arr.set_UVC([0.2 * np.cos(slam_est_th)], [0.2 * np.sin(slam_est_th)])
            
            vx_phys, vz_phys = r_state[4], r_state[5]
            vel_arr.set_offsets([[slam_u, slam_v]])
            vel_arr.set_UVC([0.2 * vx_phys * self.env._slam_scale], [0.2 * vz_phys * self.env._slam_scale])
            
            dx_believed = tgt_u_f - slam_est_u
            dy_believed = tgt_v_f - slam_est_v
            dist_believed = np.sqrt(dx_believed**2 + dy_believed**2) + 1e-8
            ux_believed = dx_believed / dist_believed
            uy_believed = dy_believed / dist_believed
            
            target_dir_arr.set_offsets([[slam_u, slam_v]])
            target_dir_arr.set_UVC([0.2 * ux_believed], [0.2 * uy_believed])
            
            # ToF beams (4 beams: -45, 0, 45, 180 deg)
            _tof_offsets = [-np.pi/4, 0.0, np.pi/4, np.pi]
            for bi, ((bl, bm), offset) in enumerate(zip(tof_beam_artists, _tof_offsets)):
                ang = sensor_heading + offset
                hu = slam_u + tof_d[bi] * np.cos(ang)
                hv = slam_v + tof_d[bi] * np.sin(ang)
                bl.set_data([slam_u, hu], [slam_v, hv])
                bm.set_data([hu], [hv])
                
            # Camera cone
            fov_r = 1.2
            fov_left_line.set_data(
                [slam_u, slam_u + fov_r * np.cos(sensor_heading - np.pi/4)],
                [slam_v, slam_v + fov_r * np.sin(sensor_heading - np.pi/4)]
            )
            fov_right_line.set_data(
                [slam_u, slam_u + fov_r * np.cos(sensor_heading + np.pi/4)],
                [slam_v, slam_v + fov_r * np.sin(sensor_heading + np.pi/4)]
            )
            
            # Highlight obstacles
            for obs_bbox, op in obs_patch_list:
                x0, y0, x1, y1 = obs_bbox
                inside = (x0 <= slam_u <= x1) and (y0 <= slam_v <= y1)
                op.set_facecolor('#cc1111' if inside else '#2a2a4c')
                op.set_edgecolor('#ff3333' if inside else '#6666bb')
                
            # Event camera strip
            vis_strip.set_data(self.sim_data['events'][frame][None, :])
            
            # Status text
            alpha_val = self.sim_data['alpha'][frame]
            surprise_val = self.sim_data['surprise'][frame]
            nav_time.set_text(
                f'θ={sensor_heading:.2f}r | ToF [{tof_d[0]:.1f}│{tof_d[1]:.1f}│{tof_d[2]:.1f}│{tof_d[3]:.1f}]m | Surprise={surprise_val:.2f} | α={alpha_val:.2f}'
            )
            
            # SOG
            occ_img.set_data(self.sim_data['sog_states'][frame].T)
            occ_img.set_clim(vmin=-0.2, vmax=1.0)
            map_robot_dot.set_data([slam_est_u], [slam_est_v])
            map_trail_line.set_data(
                [p[0] for p in slam_est],
                [p[1] for p in slam_est]
            )
            
            map_target_dir_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_target_dir_arr.set_UVC([0.2 * ux_believed], [0.2 * uy_believed])
            
            f_scale = 6.25
            fr_x, fr_y = self.sim_data['f_repel_vec'][frame]
            map_f_repel_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_f_repel_arr.set_UVC([f_scale * fr_x], [f_scale * fr_y])
            
            fb_x, fb_y = self.sim_data['f_brain_vec'][frame]
            map_f_brain_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_f_brain_arr.set_UVC([f_scale * fb_x], [f_scale * fb_y])
            
            fe_x, fe_y = self.sim_data['f_emd_vec'][frame]
            map_f_emd_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_f_emd_arr.set_UVC([f_scale * fe_x], [f_scale * fe_y])
            
            fi_x, fi_y = self.sim_data['f_instar_vec'][frame]
            map_f_instar_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_f_instar_arr.set_UVC([f_scale * fi_x], [f_scale * fi_y])
            
            fn_x, fn_y = self.sim_data['f_net_vec'][frame]
            map_f_net_arr.set_offsets([[slam_est_u, slam_est_v]])
            map_f_net_arr.set_UVC([f_scale * fn_x], [f_scale * fn_y])
            
            _tof_offsets = [-np.pi/4, 0.0, np.pi/4, np.pi]
            for bi, ((bl, bm), offset) in enumerate(zip(map_tof_beam_artists, _tof_offsets)):
                ang = slam_est_th + offset
                hu = slam_est_u + tof_d[bi] * np.cos(ang)
                hv = slam_est_v + tof_d[bi] * np.sin(ang)
                bl.set_data([slam_est_u, hu], [slam_est_v, hv])
                bm.set_data([hu], [hv])
                
            map_fov_left_line.set_data(
                [slam_est_u, slam_est_u + fov_r * np.cos(slam_est_th - np.pi/4)],
                [slam_est_v, slam_est_v + fov_r * np.sin(slam_est_th - np.pi/4)]
            )
            map_fov_right_line.set_data(
                [slam_est_u, slam_est_u + fov_r * np.cos(slam_est_th + np.pi/4)],
                [slam_est_v, slam_est_v + fov_r * np.sin(slam_est_th + np.pi/4)]
            )
            
            # Wing
            rx, rz = r_state[0], r_state[1]
            r_th, r_phi = r_state[2], r_state[3]
            ax_wing.set_xlim(rx - 0.25, rx + 0.25)
            ax_wing.set_ylim(rz - 0.25, rz + 0.25)
            
            for obs_phys, op_wing in obs_patch_wing_list:
                px0, py0, px1, py1 = obs_phys
                inside = (px0 <= rx <= px1) and (py0 <= rz <= py1)
                op_wing.set_facecolor('#cc1111' if inside else '#2a2a4c')
                op_wing.set_edgecolor('#ff3333' if inside else '#6666bb')
                
            active_props_batch = self.state[3]
            real_props = jax.tree.map(lambda x: x[0], active_props_batch)
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
            
            prev_idx = max(0, frame - 1)
            prev2_idx = max(0, frame - 2)
            
            # 1. Retrieve current and previous states
            r_state_prev1 = self.sim_data['states'][prev_idx]
            r_state_prev2 = self.sim_data['states'][prev2_idx]
            
            rx_curr, rz_curr, r_th_curr = r_state[0], r_state[1], r_state[2]
            
            # Project prev2 wing pose to current coordinates
            rx_p2, rz_p2, r_th_p2 = r_state_prev2[0], r_state_prev2[1], r_state_prev2[2]
            wx_p2, wz_p2, wang_p2 = self.sim_data['wing_pose'][prev2_idx]
            
            dx2 = wx_p2 - rx_p2
            dz2 = wz_p2 - rz_p2
            cos2, sin2 = np.cos(r_th_p2), np.sin(r_th_p2)
            dx_body2 = dx2 * cos2 + dz2 * sin2
            dz_body2 = -dx2 * sin2 + dz2 * cos2
            wang_body2 = wang_p2 - r_th_p2
            
            cos_c, sin_c = np.cos(r_th_curr), np.sin(r_th_curr)
            dx_proj2 = dx_body2 * cos_c - dz_body2 * sin_c
            dz_proj2 = dx_body2 * sin_c + dz_body2 * cos_c
            wx_proj_prev2 = rx_curr + dx_proj2
            wz_proj_prev2 = rz_curr + dz_proj2
            wang_proj_prev2 = r_th_curr + wang_body2
            
            # Project prev1 wing pose to current coordinates
            rx_p1, rz_p1, r_th_p1 = r_state_prev1[0], r_state_prev1[1], r_state_prev1[2]
            wx_p1, wz_p1, wang_p1 = self.sim_data['wing_pose'][prev_idx]
            
            dx1 = wx_p1 - rx_p1
            dz1 = wz_p1 - rz_p1
            cos1, sin1 = np.cos(r_th_p1), np.sin(r_th_p1)
            dx_body1 = dx1 * cos1 + dz1 * sin1
            dz_body1 = -dx1 * sin1 + dz1 * cos1
            wang_body1 = wang_p1 - r_th_p1
            
            dx_proj1 = dx_body1 * cos_c - dz_body1 * sin_c
            dz_proj1 = dx_body1 * sin_c + dz_body1 * cos_c
            wx_proj_prev1 = rx_curr + dx_proj1
            wz_proj_prev1 = rz_curr + dz_proj1
            wang_proj_prev1 = r_th_curr + wang_body1
            
            # Current wing pose
            wx_proj_curr, wz_proj_curr, wang_proj_curr = self.sim_data['wing_pose'][frame]
            
            # 2. Interpolate 9 poses over 0.5 cycles (last 2 control steps)
            num_w = len(wing_lines)
            for k in range(num_w):
                tau = k / float(num_w - 1) if num_w > 1 else 1.0
                if tau <= 0.5:
                    alpha = 2.0 * tau
                    wx_interp = (1.0 - alpha) * wx_proj_prev2 + alpha * wx_proj_prev1
                    wz_interp = (1.0 - alpha) * wz_proj_prev2 + alpha * wz_proj_prev1
                    
                    d_theta = wang_proj_prev1 - wang_proj_prev2
                    d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
                    wang_interp = wang_proj_prev2 + alpha * d_theta
                else:
                    alpha = 2.0 * (tau - 0.5)
                    wx_interp = (1.0 - alpha) * wx_proj_prev1 + alpha * wx_proj_curr
                    wz_interp = (1.0 - alpha) * wz_proj_prev1 + alpha * wz_proj_curr
                    
                    d_theta = wang_proj_curr - wang_proj_prev1
                    d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
                    wang_interp = wang_proj_prev1 + alpha * d_theta
                
                wing_len = self.env.phys.fluid.WING_LEN
                N_pts    = self.env.phys.fluid.N_PTS
                x_local  = np.linspace(wing_len/2, -wing_len/2, N_pts)
                c_w, s_w = np.cos(wang_interp), np.sin(wang_interp)
                wing_x   = wx_interp + x_local * c_w
                wing_z   = wz_interp + x_local * s_w
                wing_lines[k].set_data(wing_x, wing_z)
                
                if k == num_w - 1:
                    patch_le.set_center((wing_x[0], wing_z[0]))
                    patch_hinge.set_center((hinge_p[0], hinge_p[1]))
                    
                    pts = np.stack([wing_x, wing_z], axis=1)
                    quiver.set_offsets(pts)
                    quiver.set_UVC(f_nodal[:, 0], f_nodal[:, 1])
                    
            time_text.set_text(f'T={t:.4f}s | Z={rz:.3f}m')
            
            # SNN Raster - append spikes up to this frame
            for neuron_id in self.sim_data['active_places'][frame]:
                anim_spike_t.append(t)
                anim_spike_idx.append(neuron_id)
            snn_spike_line.set_data(anim_spike_t, anim_spike_idx)
            snn_time_line.set_xdata([t])
            
            # Telemetry
            time_series = self.sim_data['t'][:frame+1]
            line_surprise.set_data(time_series, self.sim_data['surprise'][:frame+1])
            line_alpha.set_data(time_series, self.sim_data['alpha'][:frame+1])
            
            f_repel_np = np.array(self.sim_data['f_repel'][:frame+1])
            line_repel.set_data(time_series, f_repel_np)
            
            f_emd_np = np.array([np.linalg.norm(v) for v in self.sim_data['f_emd_vec'][:frame+1]])
            line_emd.set_data(time_series, f_emd_np)
            
            f_instar_np = np.array([np.linalg.norm(v) for v in self.sim_data['f_instar_vec'][:frame+1]])
            line_instar.set_data(time_series, f_instar_np)
            
            max_val = 0.30
            ax_telemetry_right.set_ylim(-0.01, max_val * 1.1)
            telemetry_time_line.set_xdata([t])
            
            fig_title = f"Dashboard - Frame: {frame}/{n_frames} | Simulator Time: {t:.2f}s"
            fig.suptitle(fig_title, color='white', fontsize=12, fontweight='bold')
            
            return (patch_thorax, patch_le, patch_hinge, traj_line, hornet_dot, slam_est_line,
                    slam_est_dot, imu_dr_line, imu_dr_dot, heading_arr, slam_heading_arr,
                    target_dir_arr, map_target_dir_arr, map_f_repel_arr, map_f_brain_arr, map_f_emd_arr, map_f_instar_arr, map_f_net_arr,
                    snn_time_line, telemetry_time_line) + tuple(wing_lines)
                    
        # Compile animation and save
        ani = animation.FuncAnimation(fig, update_frame, frames=n_frames, interval=20, blit=False)
        ani.save(self.save_gif_path, writer='pillow', fps=60, dpi=60)
        plt.close(fig)
        print(f"--> Saved Interactive Inference GIF: {self.save_gif_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Flight Inference Dashboard for Embodied Hornet")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the SHAC params checkpoint file. Defaults to latest in checkpoints_shac/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_shac",
                        help="Directory to look for checkpoints if --checkpoint is not specified")
    parser.add_argument("--save-gif", type=str, default=None,
                        help="Path to save the flight history GIF. Defaults to <ckpt_dir>/inference_interactive.gif")
    parser.add_argument("--draw-skip", type=int, default=5,
                        help="Number of control steps to skip between dashboard plot redraws (default: 5)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of steps to run before auto-exiting (default: run indefinitely)")
    parser.add_argument("--gpu", action="store_true", help="Enable JAX GPU support (macOS Metal is not supported)")
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = get_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            print(f"❌ Error: No checkpoints found in '{args.checkpoint_dir}'. Please train first or supply --checkpoint.")
            sys.exit(1)
            
    # Resolve save gif path
    gif_path = args.save_gif
    if gif_path is None:
        gif_path = os.path.join(os.path.dirname(ckpt_path), "inference_interactive.gif")
        
    dashboard = InteractiveInference(
        checkpoint_path=ckpt_path,
        save_gif_path=gif_path,
        draw_skip=args.draw_skip,
        max_steps=args.steps
    )
    dashboard.run()
