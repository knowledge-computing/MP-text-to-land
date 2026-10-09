"""Subdivision (plat) matcher: NER subdivision strings -> one county plat polygon.

names.py    string normalization (step 1: upper/punct; step 3: abbreviation/ordinal classes learned
            from plat-layer name/alias pairs)
match.py    PlatIndex + MatchConfig: top-1 plat for a deed under a configuration (scorer, county restriction,
            normalization level, aliases, PLSS section hard filter or soft bonus)
geometry.py deed-vs-plat geometric metrics (contains, intersects, centroid-in, coverage, IoU, distance)
"""
from .geometry import geom_metrics  # noqa: F401
from .match import MatchConfig, PlatIndex  # noqa: F401
from .names import NameNormalizer, learn_rules, n1  # noqa: F401
