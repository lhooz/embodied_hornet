# 🐝 Embodied Hornet

**Unified Spiking SLAM & Neuromechanical Flight Control System**

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
Event Camera → [neuro-symbolic-slam] → Spatial Belief (3-DOF) + Visual Features (512-dim)
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
├── neuro-symbolic-slam/                <-- git submodule (SLAM perception, unmodified)
├── docs/
│   └── system_integration_report.md   # Full architectural specification
├── pyproject.toml
└── README.md
```

---

## 🚀 Setup

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/lhooz/embodied_hornet.git
cd embodied_hornet
```

> **Note:** The `neuro-symbolic-slam` submodule contains large binary files. If the clone stalls, run:
> ```bash
> git submodule update --init --reference /path/to/local/neuro-symbolic-slam neuro-symbolic-slam
> # or shallow:
> git submodule update --init --depth 1 neuro-symbolic-slam
> ```

### 2. Install dependencies

```bash
pip install -e .
# or use the shared workspace venv:
# source ../.venv/bin/activate
```

### 3. Run training

```bash
JAX_PLATFORMS=cpu python -m embodied_hornet.train
```

---

## Key Integration Points

All integration code lives in `embodied_hornet/` (this package). The three dependency repos are used **unmodified**:

| Module | File | What it adds |
|:---|:---|:---|
| `env.py` | [embodied_hornet/env.py](embodied_hornet/env.py) | `FlyEnv`: 1m×1m arena geometry generation (`regenerate_arena()`), SLAM coordinate mapping (hornet ±0.5m → 10m SLAM space), `compute_slam_sensors()` producing real event-camera + ToF + kinematic odometry for `SNNSLAMSystem`; `ingest_perceptual_streams()` Asymmetric Instar rule routing visual belief → 4-dim CPG input |
| `neural_idapbc.py` | [embodied_hornet/neural_idapbc.py](embodied_hornet/neural_idapbc.py) | `IDA_PBC_Hover`, `hover_stable()`, `differentiable_attention_gate()` (DNAG) — blends policy with passivity-preserving hover modulations gated by real SLAM surprise |
| `train.py` | [embodied_hornet/train.py](embodied_hornet/train.py) | Unified 12-dim observation, real `SNNSLAMSystem` integration (outside-JIT async loop), per-episode arena + SLAM reset, SHAC+PBT training loop |
| `snn_live_slam.py` | [embodied_hornet/snn_live_slam.py](embodied_hornet/snn_live_slam.py) | Thin re-export wrapper over `neuro-symbolic-slam`'s module; adds surprise telemetry logging (threshold crossings for DNAG diagnostics) |

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
