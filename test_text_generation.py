
import pytest
import xarray as xr
import numpy as np
import pandas as pd
from dashboard import _generate_explanation

def create_mock_da(var_name="P"):
    # Create a mock DataArray
    # Shape: (48, 10, 20) (4 years monthly, small spatial grid)
    times = pd.date_range("2019-01-01", periods=48, freq="ME")
    lats = np.linspace(30, 31, 10)
    lons = np.linspace(35, 36, 20)

    # Generate data:
    # West (lons < 35.5) has higher values
    data = np.zeros((48, 10, 20))
    for i, lon in enumerate(lons):
        if lon < 35.5:
            base = 50.0 # High
        else:
            base = 10.0 # Low
        data[:, :, i] = base + np.random.rand(48, 10) * 5

    # Seasonality: Max in Jan (idx 0), Min in July (idx 6)
    # Add seasonal signal
    months = times.month.to_numpy()
    season_factor = np.cos((months - 1) * 2 * np.pi / 12) + 1 # 0 to 2
    data = data * (season_factor[:, None, None] + 0.5)

    da = xr.DataArray(
        data,
        coords={"time": times, "latitude": lats, "longitude": lons},
        dims=("time", "latitude", "longitude"),
        name=var_name
    )
    return da

def test_p_explanation():
    da = create_mock_da("P")
    text = _generate_explanation("P", "TestBasin", 2019, 2022, da)

    print("\nGenerated P Text:\n", text)

    assert "spatial distribution of rainfall" in text
    assert "higher values" in text
    assert "western part" in text
    assert "below average" in text
    assert "January" in text # Peak month
    assert "lowest values occur in July" in text or "lowest values occur in June" in text # Low month

def test_et_explanation():
    da = create_mock_da("ET")
    text = _generate_explanation("ET", "TestBasin", 2019, 2022, da)

    print("\nGenerated ET Text:\n", text)

    assert "spatial distribution of ET" in text
    assert "western part" in text

def test_pet_explanation():
    da = create_mock_da("P-ET")
    text = _generate_explanation("P-ET", "TestBasin", 2019, 2022, da)

    print("\nGenerated P-ET Text:\n", text)

    assert "spatial distribution of water balance" in text
