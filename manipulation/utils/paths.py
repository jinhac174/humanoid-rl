"""Project paths. Single source of truth for filesystem locations.

All non-code resources (USDs, meshes, scenes, generated assets) live
under ASSET_ROOT. Import from here instead of recomputing parents[N]
from each callsite -- that pattern is fragile to directory changes.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
