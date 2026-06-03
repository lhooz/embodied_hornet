"""
embodied_hornet — Unified Spiking SLAM & Neuromechanical Flight Control
Integrates neuro-symbolic-slam, hornetRL, and fly_surrogate.
"""
import os
import sys
import importlib

# Force CPU only on Apple Silicon (Metal backend is unimplemented).
# On other platforms (Linux/Colab with CUDA), leave JAX_PLATFORMS alone
# so JAX auto-detects the GPU.
import platform as _platform
if _platform.system() == "Darwin" and _platform.machine() == "arm64":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Resolve the project root (embodied_hornet/)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add submodule paths so we can import their packages directly.
# These are git submodules cloned inside the embodied_hornet/ project.
# If running from the monorepo workspace, the sibling repos work too.
_submodule_paths = [
    os.path.join(_project_root, 'hornetRL'),
    os.path.join(_project_root, 'neuro-symbolic-slam', 'src'),
    os.path.join(_project_root, 'fly_surrogate'),
]

# Fallback: if submodules aren't present, try workspace siblings
_workspace_root = os.path.abspath(os.path.join(_project_root, '..'))
_sibling_paths = [
    os.path.join(_workspace_root, 'hornetRL'),
    os.path.join(_workspace_root, 'neuro-symbolic-slam', 'src'),
    os.path.join(_workspace_root, 'fly_surrogate'),
]

for _sub, _sib in zip(_submodule_paths, _sibling_paths):
    _chosen = _sub if os.path.isdir(_sub) else _sib
    if _chosen not in sys.path:
        sys.path.insert(0, _chosen)
    # Clear any stale directory listing cached in path_importer_cache
    sys.path_importer_cache.pop(_chosen, None)

# Invalidate cached module lookups so newly-added paths take effect
# even if a previous import attempt failed (e.g. during pip install).
sys.path_importer_cache.clear()
importlib.invalidate_caches()
