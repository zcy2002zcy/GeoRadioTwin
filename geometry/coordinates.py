"""Coordinate conversion for UrbanRadio3D rasters."""

from __future__ import annotations


def physical_xy_to_row_col(x: float, y: float, height: int) -> tuple[float, float]:
    """Map lower-left physical (X, Y) to top-left raster (row, column)."""

    return (height - 1) - x, y


def row_col_to_physical_xy(row: float, col: float, height: int) -> tuple[float, float]:
    """Map top-left raster (row, column) to lower-left physical (X, Y)."""

    return (height - 1) - row, col

