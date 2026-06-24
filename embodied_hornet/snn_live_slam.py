"""
embodied_hornet/snn_live_slam.py

Thin integration wrapper over neuro-symbolic-slam's snn_live_slam module.
Re-exports everything from the submodule unchanged, and adds:
  - Surprise telemetry logging (threshold-crossing printouts for DNAG diagnostics)

Usage:
    from embodied_hornet.snn_live_slam import run_live_slam_with_telemetry

Note: Actual DNAG gating runs inside embodied_hornet/train.py's loss_fn.
      The telemetry here is for diagnostic/monitoring purposes only.
"""

# Re-export everything from the submodule so downstream code can import
# from here as if it were the original module.
from snn_live_slam import *                          # noqa: F401, F403
from snn_live_slam import run_live_slam as _run_live_slam_original

# ==============================================================================
# INTEGRATION ADDITION: SURPRISE TELEMETRY PATCH
# ==============================================================================
# We wrap the original run_live_slam to inject threshold-crossing printouts
# after each `surprise` update. This avoids duplicating 1329 lines.

import functools


def _make_telemetry_step(original_forward_step):
    """
    Wraps a system's forward_step to emit surprise telemetry after each call.
    """
    @functools.wraps(original_forward_step)
    def _patched_forward_step(*args, **kwargs):
        result = original_forward_step(*args, **kwargs)
        pose_cl, r_place, r_ring, is_confident, peak_idx_place, debug_gates = result

        # Compute surprise from debug gates (same formula as original)
        raw_match = float(debug_gates['Raw_Match'][0])
        conc_place = float(debug_gates['Conc_Place'][0])
        composite_match = raw_match
        import numpy as np
        surprise = float(1.0 - np.exp(-5.0 * (1.0 - composite_match)))

        # 🌟 SURPRISE TELEMETRY FOR FLIGHT CONTROLLER AWARENESS 🌟
        if surprise >= 0.30:
            print(f"\n⚠️  [SURPRISE TELEMETRY] S={surprise:.2f} >= 0.30 — hover gating would activate.")
        if surprise >= 0.60:
            print(f"\n🔒 [SURPRISE TELEMETRY] S={surprise:.2f} >= 0.60 — STDP plasticity locked.")

        return result
    return _patched_forward_step


def run_live_slam_with_telemetry(*args, **kwargs):
    """
    Runs the standard live SLAM loop with surprise telemetry logging enabled.
    Accepts the same arguments as snn_live_slam.run_live_slam().

    Surprise thresholds logged:
      >= 0.30 → DNAG hover gating would activate (in embodied_hornet/train.py)
      >= 0.60 → STDP plasticity freeze would activate (in neuro-symbolic-slam)
    """
    # Dynamically patch the system class used inside run_live_slam.
    # We import the system class, patch its forward_step, then restore.
    import snn_slam_system as _sys_mod

    original_forward = _sys_mod.SNNSLAMSystem.forward_step
    _sys_mod.SNNSLAMSystem.forward_step = _make_telemetry_step(original_forward)

    try:
        return _run_live_slam_original(*args, **kwargs)
    finally:
        # Always restore original, even on exception
        _sys_mod.SNNSLAMSystem.forward_step = original_forward