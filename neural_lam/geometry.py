"""Geometry and spherical coordinate conversion utilities."""

# Third-party
import numpy as np
import torch


def latlon_to_cartesian(
    lat: np.ndarray | torch.Tensor,
    lon: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Convert latitude/longitude coordinates (in degrees) to 3D Cartesian
    (x, y, z) coordinates on a unit sphere.

    Parameters
    ----------
    lat : np.ndarray or torch.Tensor
        Latitude coordinates in degrees.
    lon : np.ndarray or torch.Tensor
        Longitude coordinates in degrees.

    Returns
    -------
    np.ndarray or torch.Tensor
        Cartesian coordinates with shape (..., 3).
    """
    if isinstance(lat, torch.Tensor):
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack((x, y, z), dim=-1)
    else:
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.stack((x, y, z), axis=-1)


def calculate_area_weights(
    lat: np.ndarray | torch.Tensor,
    grid_type: str = "equiangular",
) -> np.ndarray | torch.Tensor:
    """
    Calculate spatial area weights for grid nodes based on latitude.

    Parameters
    ----------
    lat : np.ndarray or torch.Tensor
        Latitude coordinates of grid nodes in degrees.
    grid_type : str, optional
        Type of the grid ("equiangular" or others). Default is "equiangular".

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized area weights (sum to 1.0) with same shape as lat.
    """
    if grid_type == "equiangular":
        if isinstance(lat, torch.Tensor):
            lat_rad = torch.deg2rad(lat)
            weights = torch.cos(lat_rad)
            return weights / torch.sum(weights)
        else:
            lat_rad = np.deg2rad(lat)
            weights = np.cos(lat_rad)
            return weights / np.sum(weights)
    else:
        # For LAM or other grids, default to uniform weights
        if isinstance(lat, torch.Tensor):
            return torch.ones_like(lat) / lat.numel()
        else:
            return np.ones_like(lat) / lat.size
