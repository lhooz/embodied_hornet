"""
embodied_hornet — Unified Spiking SLAM & Neuromechanical Flight Control
Integrates neuro-symbolic-slam, hornetRL, and fly_surrogate.
"""
import os
import sys

# Force CPU for Apple Silicon compatibility (Metal UNIMPLEMENTED memory space)
os.environ["JAX_PLATFORMS"] = "cpu"

# Add sibling repos to Python path so we can import their modules
_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_sibling_paths = [
    os.path.join(_workspace, 'hornetRL'),
    os.path.join(_workspace, 'neuro-symbolic-slam', 'src'),
    os.path.join(_workspace, 'fly_surrogate'),
]
for _p in _sibling_paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
