# 🐝 Embodied Hornet

**Unified Spiking SLAM & Neuromechanical Flight Control System**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhooz/embodied_hornet/blob/main/notebooks/demo_colab.ipynb)

This project integrates three independent subsystems into a single JAX-accelerated pipeline for autonomous insect-scale flight with neuromorphic spatial intelligence.

---

## Architecture

```
embodied_hornet (this project — integration layer)
├── neuro-symbolic-slam    → Perception & Mapping (Spiking CANN, STDP, Place Cells)
├── hornetRL               → Differentiable Flight Control (IDA-PBC, Neural CPG)
└── fly_surrogate          → Aerodynamic Physics (LBM Surrogates, ResNet Forces)
```

### System Flow

```
IMU [Acc, Gyro] → [Complementary Filter] ──┐
                                           ↓
Event Camera ───→ [neuro-symbolic-slam] ──→ Spatial Belief (3-DOF) + Visual Features (512-dim)
                                                       ↓
                                             Asymmetric Instar Rule
                                                       ↓
                                   [hornetRL] ← Weighted Perceptual Belief (4-dim)
                                                       ↓
                                             IDA-PBC + Neural CPG → Wing Kinematics
                                                       ↓
                                             [fly_surrogate] → Aerodynamic Forces
                                                       ↓
                                             Port-Hamiltonian Dynamics → Next State
```

---

## 📂 Project Structure

```text
embodied_hornet/                        <-- This repository (integration layer)
├── embodied_hornet/                    <-- Python integration package
│   ├── __init__.py                     # sys.path setup for submodule dependency resolution
│   ├── train.py                        # Unified SHAC+PBT training loop; real SNNSLAMSystem
│   │                                   #   integration (outside-JIT, async multi-rate)
│   ├── env.py                          # FlyEnv: Port-Hamiltonian flight physics +
│   │                                   #   arena generation, SLAM sensor pipeline,
│   │                                   #   Asymmetric Instar perceptual routing
│   ├── neural_idapbc.py                # IDA-PBC energy shaping, hover_stable(),
│   │                                   #   differentiable attention gate (DNAG)
│   └── snn_live_slam.py                # Thin re-export wrapper + surprise telemetry
│                                       #   logging for DNAG diagnostics
├── hornetRL/                           <-- git submodule (base flight control, unmodified)
├── fly_surrogate/                      <-- git submodule (aerodynamic physics, unmodified)
├── neuro-symbolic-slam/                <-- git submodule (SLAM perception, modified to add gravity fusion)
├── notebooks/
│   └── demo_colab.ipynb               # Google Colab demo (see badge above)
├── docs/
│   └── system_integration_report.md   # Full architectural specification
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### Run on Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhooz/embodied_hornet/blob/main/notebooks/demo_colab.ipynb)

Click the badge to open the demo notebook. It will:
1. Mount your Drive for persistent checkpoints
2. Clone the repo with all submodules in one command
3. Skip hover-specialist training (pre-trained `hover_params.pkl` is included)
4. Launch GPU-accelerated SHAC navigation training
5. Display epoch GIFs with the SLAM sensor overlay (ToF beams, FOV cone, collision indicators)

> ⚠️ **Troubleshooting:** If you encounter `ModuleNotFoundError: No module named 'sparse_forest'` or other import errors during Cell 2 verification after cloning or pulling updates:
> 1. Run **Cell 1** to ensure submodules are fully checked out.
> 2. Select **Runtime → Restart runtime** from the top menu to clear the Python kernel import caches.
> 3. Rerun all cells.


### Local Setup

#### 1. Clone with submodules

```bash
git clone --recursive https://github.com/lhooz/embodied_hornet.git
cd embodied_hornet
```

> **Note:** The `neuro-symbolic-slam` submodule contains large binary files. If the clone stalls, run:
> ```bash
> git submodule update --init --depth 1 neuro-symbolic-slam
> ```

#### 2. Install dependencies

```bash
pip install -e .
```

> **Note:** There is no need to run `pip install -e` on the individual submodules. The `embodied_hornet` integration package automatically registers all submodule paths (such as `neuro-symbolic-slam/src`) in `sys.path` at runtime.


#### 3. Run training

```bash
# CPU (Apple Silicon / no GPU)
python -m embodied_hornet.train

# GPU (CUDA)
python -m embodied_hornet.train --gpu
```

---

## Key Integration Points

All integration code lives in `embodied_hornet/` (this package). The dependency repos (hornetRL, fly_surrogate) are used **unmodified**, while neuro-symbolic-slam has been modified to incorporate gravity correction features:

| Module | File | What it adds |
|:---|:---|:---|
| `env.py` | [embodied_hornet/env.py](embodied_hornet/env.py) | `FlyEnv`: 1m×1m arena geometry generation (`regenerate_arena()`), SLAM coordinate mapping (hornet ±0.5m → 10m SLAM space), `compute_slam_sensors()` producing real event-camera + ToF + kinematic + MEMS accelerometer odometry (with physical flapping vibrations) for `SNNSLAMSystem`; `ingest_perceptual_streams()` Asymmetric Instar rule routing visual belief → 4-dim CPG input |
| `neural_idapbc.py` | [embodied_hornet/neural_idapbc.py](embodied_hornet/neural_idapbc.py) | `IDA_PBC_Hover`, `hover_stable()`, `differentiable_attention_gate()` (DNAG) — blends policy with passivity-preserving hover modulations gated by real SLAM surprise |
| `train.py` | [embodied_hornet/train.py](embodied_hornet/train.py) | Unified 12-dim observation, real `SNNSLAMSystem` integration (outside-JIT async loop), high-frequency decimation/accumulation of accelerometer signals, per-episode arena + SLAM reset, SHAC+PBT training loop |
| `snn_live_slam.py` | [embodied_hornet/snn_live_slam.py](embodied_hornet/snn_live_slam.py) | Thin re-export wrapper over `neuro-symbolic-slam`'s module; adds surprise telemetry logging (threshold crossings for DNAG diagnostics) |
| `snn_pose_cann.py` | [neuro-symbolic-slam/src/snn_pose_cann.py](neuro-symbolic-slam/src/snn_pose_cann.py) | `PoseCANN` heading ring attractor accepting estimated `theta_gravity` to inject corrective Gaussian currents, pulling the belief bump into alignment to arrest yaw drift. |
| `snn_slam_system.py` | [neuro-symbolic-slam/src/snn_slam_system.py](neuro-symbolic-slam/src/snn_slam_system.py) | `SNNSLAMSystem.forward_step` complementary filter fusing proper acceleration and gyroscope rates to estimate absolute gravity pitch. |

---

## 🧠 Neuromorphic Obstacle Avoidance

This project incorporates a dual-pathway, neuromorphic obstacle avoidance system inspired by biological flying insects. It combines low-latency reflexive steering with topological spatial maps to safely navigate complex environments:

### Hybrid Biological Stream Architecture
1. **The Dorsal Stream (LPTC Optic Flow Reflex):** A 32-pixel ommatidial array (downsampled from the 256-pixel event camera) feeds into a **Hassenstein-Reichardt cross-correlator** — the canonical insect Elementary Motion Detector (EMD). Adjacent pixel pairs are temporally delayed and multiplied to extract signed local motion, then pooled into two wide-field **Lobula Plate Tangential Cell (LPTC)** reflexes:
   - **HS-cell centering:** Left-vs-right flow differential drives a pitch torque correction (optomotor centering response).
   - **LGMD looming escape:** Total unsigned flow energy triggers forward deceleration when rapid visual expansion is detected.
2. **The Ventral Stream (Central Complex Map Navigation):** Uses the Spiking Occupancy Grid (SOG) representing local memory to construct a fully differentiable **Artificial Potential Field (APF)**. The gradient of this field is added directly to the port-Hamiltonian energy function, generating Lyapunov-stable steering forces away from obstacles.

```mermaid
graph TD
    EventCam["1D Ommatidial Array (32 pix)"] --> Reichardt["Hassenstein-Reichardt EMD<br/>(Delay-and-Correlate)"]
    ToF["3-Beam ToF Sonar"] --> SOG["Spiking Occupancy Grid Map (LIF Sheet)"]
    
    Reichardt --> HS["HS-Cell Pool<br/>(L-R Flow Differential)"]
    Reichardt --> LGMD["LGMD Pool<br/>(Total Expansion Energy)"]
    
    HS -->|Centering torque| IDAPBC["Neural IDA-PBC Actor"]
    LGMD -->|Braking force| IDAPBC
    SOG -->|Repulsive potential| APF["Artificial Potential Field (Ventral)"]
    APF -->|Energy gradient| IDAPBC
    
    IDAPBC --> CPG["CPG Modulator"]
    CPG --> Kinematics["Wing Kinematics"]
```

---

## Dependencies

| Repo | Role | Linked as |
|:---|:---|:---|
| [neuro-symbolic-slam](https://github.com/lhooz/neuro-symbolic-slam) | Spiking CANN pose tracking, STDP vision, HDC place cells | git submodule |
| [hornetRL](https://github.com/lhooz/hornetRL) | Port-Hamiltonian flight controller, ICNN energy shaping, spiking CPG | git submodule |
| [fly_surrogate](https://github.com/lhooz/fly_surrogate) | Differentiable aerodynamic surrogate (Taichi LBM fluid solver) | git submodule |

---

## Technical Stack

- **Framework:** JAX (functional, XLA-compiled, auto-differentiable)
- **Neural:** dm-haiku (ICNN, Critic networks)
- **Optimization:** optax (SHAC + PBT)
- **Physics:** Port-Hamiltonian rigid body dynamics + differentiable ResNet fluid surrogates
- **Perception:** Spiking neural networks (CANN, Ring Attractor, STDP, CSNN)

---

## Reference

See [docs/system_integration_report.md](docs/system_integration_report.md) for the full architectural specification.
