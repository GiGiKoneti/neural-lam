import numpy as np
import torch
import pytest

from neural_lam.geometry import latlon_to_cartesian, calculate_area_weights
from tests.dummy_datastore import DummyDatastore


def test_latlon_to_cartesian_numpy():
    """Test latlon_to_cartesian with numpy arrays."""
    # Equator, prime meridian
    lat = np.array([0.0, 90.0, 0.0])
    lon = np.array([0.0, 0.0, 90.0])
    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    res = latlon_to_cartesian(lat, lon)
    assert np.allclose(res, expected, atol=1e-6)


def test_latlon_to_cartesian_torch():
    """Test latlon_to_cartesian with torch tensors."""
    lat = torch.tensor([0.0, 90.0, 0.0])
    lon = torch.tensor([0.0, 0.0, 90.0])
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    res = latlon_to_cartesian(lat, lon)
    assert torch.allclose(res, expected, atol=1e-6)


def test_calculate_area_weights_equiangular():
    """Test calculate_area_weights with equiangular grid type."""
    # Latitude values from Equator to 60 degrees
    lat = np.array([0.0, 30.0, 60.0])
    weights = calculate_area_weights(lat, grid_type="equiangular")
    
    # Weights at equator must be larger than at 60 degrees (cos(0) > cos(60))
    assert weights[0] > weights[2]
    assert np.isclose(np.sum(weights), 1.0)

    # Test with torch tensor
    lat_t = torch.tensor([0.0, 30.0, 60.0])
    weights_t = calculate_area_weights(lat_t, grid_type="equiangular")
    assert weights_t[0] > weights_t[2]
    assert torch.isclose(torch.sum(weights_t), torch.tensor(1.0))


def test_calculate_area_weights_uniform():
    """Test calculate_area_weights with uniform grid type."""
    lat = np.array([0.0, 30.0, 60.0])
    weights = calculate_area_weights(lat, grid_type="uniform")
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(weights, expected)


def test_datastore_get_area_weights():
    """Test get_area_weights on a BaseDatastore instance."""
    datastore = DummyDatastore(n_grid_points=100)
    weights = datastore.get_area_weights("state")
    
    # Since DummyDatastore does not use PlateCarree projection,
    # it should return uniform weights.
    assert weights.shape == (100,)
    assert np.allclose(weights, np.ones(100) / 100)
    assert np.isclose(np.sum(weights), 1.0)
