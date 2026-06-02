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

## Dependencies (Sibling Repos)

| Repo | Path | Description |
|:---|:---|:---|
| [neuro-symbolic-slam](https://github.com/lhooz/neuro-symbolic-slam) | `../neuro-symbolic-slam/` | Spiking neural SLAM with CANN pose tracking, STDP vision, HDC place cells |
| [hornetRL](https://github.com/lhooz/hornetRL) | `../hornetRL/` | Port-Hamiltonian flight controller with ICNN energy shaping and spiking CPG |
| [fly_surrogate](https://github.com/lhooz/fly_surrogate) | `../fly_surrogate/` | Differentiable aerodynamic surrogate trained from Taichi LBM fluid solver |

## Workspace Layout

```
workspace/
├── neuro-symbolic-slam/   ← git repo (perception & mapping)
├── hornetRL/              ← git repo (flight control)
├── fly_surrogate/         ← git repo (aerodynamic physics)
├── embodied_hornet/       ← THIS PROJECT (integration layer)
│   ├── configs/           ← Integration-specific configuration
│   ├── docs/              ← Architectural specs & reports
│   └── README.md
└── .venv/                 ← Shared Python environment
```

## Key Integration Points

1. **Perception → Control routing** via Asymmetric Instar rule in `hornetRL/hornetRL/env.py`
2. **Surprise-driven gating** via DNAG in `hornetRL/hornetRL/neural_idapbc.py`
3. **Surprise telemetry** from `neuro-symbolic-slam/src/snn_live_slam.py`
4. **Unified SHAC training** with visual surprise in `hornetRL/hornetRL/train.py`

## Technical Stack

- **Framework:** JAX (functional, XLA-compiled, auto-differentiable)
- **Neural:** dm-haiku (ICNN, Critic networks)
- **Optimization:** optax (SHAC + PBT)
- **Physics:** Port-Hamiltonian rigid body dynamics + differentiable ResNet fluid surrogates
- **Perception:** Spiking neural networks (CANN, Ring Attractor, STDP, CSNN)

## Reference

See [docs/system_integration_report.md](docs/system_integration_report.md) for the full architectural specification.
