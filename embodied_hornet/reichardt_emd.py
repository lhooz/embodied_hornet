"""
Hassenstein-Reichardt Elementary Motion Detector (EMD) — Dorsal Visual Stream
=============================================================================

Bio-faithful implementation of the insect lobula plate tangential cell (LPTC)
optic flow pathway for reflexive obstacle avoidance.

Processing Pipeline (mirrors insect visual system):
    Retina (ommatidial array) → Lamina (temporal high-pass) →
    Medulla (delay τ) → Lobula Plate (Reichardt cross-correlate) →
    LPTCs (wide-field pool: HS centering + LGMD looming)

Two output channels:
    1. HS-cell centering reflex: left-vs-right flow differential → pitch torque
    2. LGMD looming escape reflex: total expansion energy → forward deceleration

References:
    - Hassenstein & Reichardt (1956). Systemtheoretische Analyse der Zeit-,
      Reihenfolgen- und Vorzeichenauswertung bei der Bewegungsperzeption
      des Rüsselkäfers Chlorophanus.
    - Borst, A. & Haag, J. (2002). Neural networks in the cockpit of the fly.
      J Comp Physiol A, 188(6), 419-437.
    - Srinivasan, M.V. (2010). Honey bees as a model for vision, perception,
      and cognition. Annu Rev Entomol, 55, 267-284.
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_EMD_PIX = 32          # Coarse ommatidial array (dorsal stream)
EMD_FOV_RAD = jnp.pi/2  # 90° field of view (same as full event camera)
EMD_TAU = 0.02          # Reichardt detector delay time constant (seconds)
EMD_DT = 0.002          # Physics timestep (matches Config.DT * SIM_SUBSTEPS)


def compute_emd_intensities(robot_pos, robot_heading, segments):
    """
    Render a coarse 1D ommatidial intensity array.
    
    Unlike the full 256-pixel event camera which samples barcode textures,
    this uses a simple distance-based intensity model (I = 1/(d+ε)) that
    mimics insect photoreceptor response to nearby surfaces.  This is
    biologically plausible — insect ommatidia primarily encode contrast
    edges and proximity, not fine texture.
    
    Args:
        robot_pos:      (2,) SLAM position [u, v] in metres
        robot_heading:  scalar, sensor heading in radians
        segments:       (S, 2, 2) wall/obstacle line segments
        
    Returns:
        intensities:    (N_EMD_PIX,) distance-based intensity array
        min_dists:      (N_EMD_PIX,) raw distances (for diagnostics)
    """
    angles = robot_heading + jnp.linspace(-EMD_FOV_RAD/2, EMD_FOV_RAD/2, N_EMD_PIX)
    origins = jnp.broadcast_to(robot_pos, (N_EMD_PIX, 2))
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    
    # Ray-cast against all segments (reuses the same cast_rays as ToF/event camera)
    dists, _ = _cast_rays(origins, dirs, segments)
    min_dists = jnp.min(dists, axis=-1)  # (N_EMD_PIX,) closest hit per pixel
    
    # Distance-based intensity: bright when close, dim when far
    # Clipped to prevent infinity at d=0
    intensities = 1.0 / (min_dists + 0.01)
    
    return intensities, min_dists


def _cast_rays(origins, directions, segments):
    """
    Vectorised ray-segment intersection (pure JAX).
    Identical to sparse_forest.cast_rays — duplicated here to avoid
    import dependency issues and keep the EMD module self-contained.
    
    Args:
        origins:    (N, 2) ray origins
        directions: (N, 2) ray direction unit vectors
        segments:   (S, 2, 2) line segments [[start_xy], [end_xy]]
    
    Returns:
        dists:    (N, S) parametric distances (1e6 if no hit)
        hit_pts:  (N, S, 2) intersection points
    """
    A = segments[:, 0, :]   # (S, 2) segment start
    B = segments[:, 1, :]   # (S, 2) segment end
    E = B - A               # (S, 2) segment edge vector
    D = directions[:, None, :]   # (N, 1, 2)
    diff = A[None, :, :] - origins[:, None, :]  # (N, S, 2)
    
    det = D[:, :, 0] * E[None, :, 1] - D[:, :, 1] * E[None, :, 0]
    safe = jnp.where(jnp.abs(det) > 1e-10, det, 1.0)
    t = (diff[:, :, 0] * E[None, :, 1] - diff[:, :, 1] * E[None, :, 0]) / safe
    s = (diff[:, :, 0] * D[:, :, 1] - diff[:, :, 1] * D[:, :, 0]) / safe
    valid = (jnp.abs(det) > 1e-10) & (t > 0.01) & (s >= 0) & (s <= 1)
    dists = jnp.where(valid, t, 1e6)
    hit_pts = origins[:, None, :] + t[:, :, None] * directions[:, None, :]
    
    return dists, hit_pts


def reichardt_correlate(prev_intensities, curr_intensities, alpha=None):
    """
    Hassenstein-Reichardt cross-correlation between adjacent ommatidial pairs.
    
    For each adjacent pair (i, i+1), computes:
        R_preferred = delay(I_i) × I_{i+1}     (rightward motion detector)
        R_null      = I_i × delay(I_{i+1})      (leftward motion detector)
        local_flow  = R_preferred - R_null       (signed local motion)
    
    The temporal delay is implemented as an exponential moving average (EMA):
        delayed(t) = α·prev(t-1) + (1-α)·curr(t)
    where α = exp(-dt/τ).  This is a causal first-order low-pass filter,
    biologically equivalent to the synaptic delay in T4/T5 neurons.
    
    Args:
        prev_intensities:  (N_EMD_PIX,) previous frame intensities
        curr_intensities:  (N_EMD_PIX,) current frame intensities
        alpha:             EMA smoothing factor (default: exp(-EMD_DT/EMD_TAU))
    
    Returns:
        local_flow:  (N_EMD_PIX-1,) signed local motion at each EMD pair
    """
    if alpha is None:
        alpha = jnp.exp(-EMD_DT / EMD_TAU)
    
    # Temporal delay via exponential moving average
    # delayed ≈ signal from the previous timestep, smoothed
    delayed = alpha * prev_intensities + (1.0 - alpha) * curr_intensities
    
    # Reichardt half-detectors for adjacent pixel pairs
    # Preferred direction: delayed left × current right (detects rightward motion)
    R_preferred = delayed[:-1] * curr_intensities[1:]
    # Null direction: current left × delayed right (detects leftward motion)
    R_null = curr_intensities[:-1] * delayed[1:]
    
    # Full Reichardt output: difference of half-detectors
    local_flow = R_preferred - R_null
    
    return local_flow


def pool_lptc(local_flow):
    """
    Pool local EMD outputs into wide-field LPTC responses.
    
    Implements two biologically distinct cell types:
    
    1. HS (Horizontal System) cells — optomotor centering:
       Computes left-vs-right flow differential.  When more approach flow 
       is detected on one side, the centering signal steers away.
       HS cells are sensitive to flow direction (signed).
       
    2. LGMD (Lobula Giant Movement Detector) — looming escape:
       Computes total unsigned flow energy across the visual field.
       When the total expansion exceeds a threshold, this triggers
       an emergency deceleration.  LGMD is sensitive to flow magnitude,
       not direction.
    
    Args:
        local_flow:  (N_EMD_PIX-1,) signed local motion from Reichardt EMDs
    
    Returns:
        signals:  (2,) array of [centering_signal, looming_signal]
    """
    n = local_flow.shape[0]
    mid = n // 2
    
    # HS-cell centering reflex: left-side mean flow vs right-side mean flow
    # Convention: positive flow = rightward motion (object approaching from right)
    # Centering = right_flow - left_flow:
    #   positive → more approach on the right → steer left (negative pitch torque)
    flow_left = jnp.mean(local_flow[:mid])
    flow_right = jnp.mean(local_flow[mid:])
    centering = flow_right - flow_left
    
    # LGMD looming escape: total unsigned flow energy
    # High values indicate rapid visual expansion from any direction
    # Apply soft rectification (ReLU-like) to emphasise strong signals
    looming = jnp.mean(jnp.abs(local_flow))
    
    return jnp.array([centering, looming])


def compute_emd_signals(prev_intensities, curr_intensities):
    """
    Full dorsal stream pipeline: intensities → EMD flow → LPTC signals.
    
    Convenience function combining Reichardt correlation and LPTC pooling.
    Designed to be vmapped over the batch dimension.
    
    Args:
        prev_intensities:  (N_EMD_PIX,) previous frame ommatidial intensities
        curr_intensities:  (N_EMD_PIX,) current frame ommatidial intensities
    
    Returns:
        signals:  (2,) [centering_signal, looming_signal]
    """
    local_flow = reichardt_correlate(prev_intensities, curr_intensities)
    signals = pool_lptc(local_flow)
    return signals
