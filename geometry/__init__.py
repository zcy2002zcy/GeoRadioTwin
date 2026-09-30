"""Propagation geometry and coordinate-convention utilities."""

from .coordinates import physical_xy_to_row_col, row_col_to_physical_xy
from .distance import euclidean_distance_3d
from .ray_query import RayQueryResult, query_propagation_geometry

__all__ = [
    "RayQueryResult",
    "euclidean_distance_3d",
    "physical_xy_to_row_col",
    "query_propagation_geometry",
    "row_col_to_physical_xy",
]
