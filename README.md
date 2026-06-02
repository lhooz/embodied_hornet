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
│   ├── __init__.py                     # sys.path setup for dependency resolution
│   ├── train.py                        # Unified SHAC+PBT training loop (12-dim obs, DNAG, Instar)
│   ├── env.py                          # FlyEnv + Asymmetric Instar perceptual routing
│   ├── neural_idapbc.py                # IDA-PBC + hover_stable + DNAG attention gate
│   └── snn_live_slam.py                # SLAM orchestrator + surprise telemetry
├── hornetRL/                           <-- git submodule (base flight control, unmodified)
├── fly_surrogate/                      <-- git submodule (aerodynamic physics, unmodified)
├── neuro-symbolic-slam/                <-- shallow clone (large binaries; see setup below)
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

### 2. Clone neuro-symbolic-slam separately

The SLAM repo contains large binary files and cannot be registered as a git submodule reliably. Clone it manually into the project root:

```bash
git clone --depth 1 https://github.com/lhooz/neuro-symbolic-slam.git
```

### 3. Install dependencies

```bash
pip install -e .
# or use the shared workspace venv:
# source ../.venv/bin/activate
```

### 4. Run training

```bash
JAX_PLATFORMS=cpu python -m embodied_hornet.train
```

---

## Key Integration Points

All integration code lives in `embodied_hornet/` (this package). The three dependency repos are used **unmodified**:

| Module | File | What it adds |
|:---|:---|:---|
| `env.py` | [embodied_hornet/env.py](embodied_hornet/env.py) | `ingest_perceptual_streams()` — Asymmetric Instar rule routing 515-dim visual belief → 4-dim CPG input |
| `neural_idapbc.py` | [embodied_hornet/neural_idapbc.py](embodied_hornet/neural_idapbc.py) | `IDA_PBC_Hover`, `hover_stable()`, `differentiable_attention_gate()` (DNAG) |
| `train.py` | [embodied_hornet/train.py](embodied_hornet/train.py) | Unified 12-dim observation, visual-similarity surprise signal, SHAC+PBT training loop |
| `snn_live_slam.py` | [embodied_hornet/snn_live_slam.py](embodied_hornet/snn_live_slam.py) | Surprise telemetry logging (threshold crossings for DNAG diagnostics) |

---

## Dependencies

| Repo | Role | Linked as |
|:---|:---|:---|
| [neuro-symbolic-slam](https://github.com/lhooz/neuro-symbolic-slam) | Spiking CANN pose tracking, STDP vision, HDC place cells | shallow clone |
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
