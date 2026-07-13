"""
sttdamage - reproducible synthetic tissue damage for spatial-transcriptomics
serial-section registration benchmarks.

Four physically-motivated damage types (tear, tissue loss, fold, stretch), each
parameterized by an integer severity level 0..5 (0 = undamaged control) and a
seed, applied to a spot point cloud with an exact per-spot ground-truth
displacement so registration error is measurable to the spot.

    from damage import apply_damage
    res = apply_damage(coords, "tear", severity=3, seed=0)
    res.coords            # damaged coordinates of surviving spots
    res.displacement      # exact ground-truth forward transform (invert to register)
    res.mask              # per-spot label: intact / removed / displaced / folded / stretched
"""
from .generators import (DAMAGE_TYPES, LABELS, N_LEVELS, SEVERITY, TEAR_PATHS,
                         DamageResult, apply_damage)

__all__ = ["apply_damage", "DamageResult", "DAMAGE_TYPES", "SEVERITY",
           "TEAR_PATHS", "LABELS", "N_LEVELS"]
__version__ = "0.1.0"
