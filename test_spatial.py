from dashboard import load_and_process_data, find_nc_file
import numpy as np
import xarray as xr
import pytest
import os

basin = "Amman Zarqa"

def test_find_nc_file():
    # Test if file finding works
    p_file = find_nc_file(basin, "P")
    assert p_file is not None, "Precipitation file not found"
    assert "P" in p_file

    et_file = find_nc_file(basin, "ET")
    assert et_file is not None, "ET file not found"
    assert "ET" in et_file

    lu_file = find_nc_file(basin, "LU")
    assert lu_file is not None, "LU file not found"
    assert "LU" in lu_file

def test_load_and_process_data():
    # Test data loading
    da, var, name = load_and_process_data(basin, "P", year_start=2019, year_end=2019)
    assert da is not None, "Failed to load P data"
    assert var is not None
    assert name is not None
    # Check if dimensions are correct (coarsened to 1km approx)
    # The dummy data size is unknown but it should be an xarray DataArray
    assert isinstance(da, xr.DataArray)
