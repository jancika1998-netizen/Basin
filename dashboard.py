import os
import glob
import numpy as np
import pandas as pd
import collections
import textwrap

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import xarray as xr
import geopandas as gpd
import fiona
from shapely.geometry import shape as shp_shape, mapping
from shapely import wkb as shp_wkb


# =========================
# XARRAY / NETCDF UTILITIES
# =========================

def _open_xr_dataset(fp: str) -> xr.Dataset:
    """Open NetCDF with engine fallback to avoid backend errors."""
    for eng in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(fp, decode_times=True, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Failed to open dataset with available engines: {fp}")

def _standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """Normalize latitude/longitude names to 'latitude'/'longitude' and ensure ascending latitude."""
    lat_names = ["latitude", "lat", "y"]
    lon_names = ["longitude", "lon", "x"]
    lat = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)
    if lat and lat != "latitude":
        ds = ds.rename({lat: "latitude"})
    if lon and lon != "longitude":
        ds = ds.rename({lon: "longitude"})
    if "latitude" in ds.dims:
        lat_vals = ds["latitude"].values
        if lat_vals.size > 1 and lat_vals[1] < lat_vals[0]:
            ds = ds.sortby("latitude")
    return ds

def _pick_data_var(ds: xr.Dataset):
    """Pick the first 2D/3D field with latitude/longitude dims."""
    exclude = {"time", "latitude", "longitude", "crs", "spatial_ref"}
    cands = [v for v in ds.data_vars if v not in exclude]
    if not cands:
        return None
    with_ll = [v for v in cands if {"latitude", "longitude"}.issubset(set(ds[v].dims))]
    return with_ll[0] if with_ll else cands[0]


# ======================
# FILE / PATH UTILITIES
# ======================

BASE_DIR = os.getcwd()
BASIN_DIR = os.path.join(BASE_DIR, "basins")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

def _first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def find_nc_file(basin_name: str, variable_type: str):
    """Find a representative NetCDF per variable type in a basin folder."""
    netcdf_dir = os.path.join(BASIN_DIR, basin_name, "NetCDF")
    if not os.path.isdir(netcdf_dir):
        return None
    if variable_type == "P":
        pats = [os.path.join(netcdf_dir, "*_P_*.nc"), os.path.join(netcdf_dir, "*P*.nc")]
    elif variable_type == "ET":
        pats = [os.path.join(netcdf_dir, "*_ETa_*.nc"), os.path.join(netcdf_dir, "*_ET_*.nc"), os.path.join(netcdf_dir, "*ET*.nc")]
    elif variable_type == "LU":
        pats = [os.path.join(netcdf_dir, "*_LU_*.nc"), os.path.join(netcdf_dir, "*LandUse*.nc"), os.path.join(netcdf_dir, "*LU*.nc")]
    else:
        return None
    return _first_existing(pats)

def find_shp_file(basin_name: str):
    shp_dir = os.path.join(BASIN_DIR, basin_name, "Shapefile")
    if not os.path.isdir(shp_dir):
        return None
    return _first_existing([os.path.join(shp_dir, "*.shp")])


# ======================
# TEXT / CONTENT UTILITIES
# ======================

def read_common_text(filename: str) -> str:
    """Read a text file from the assets directory."""
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading {filename}: {e}"
    return f"File {filename} not found."

def read_basin_text(basin_name: str, filename: str) -> str:
    """Read a text file from the basin directory."""
    # Check root of basin folder first (for lu.txt, study area.txt)
    path = os.path.join(BASIN_DIR, basin_name, filename)
    if os.path.exists(path):
         try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
         except Exception:
            pass

    # Fallback to text/ subdirectory
    path = os.path.join(BASIN_DIR, basin_name, "text", filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            pass

    return f"No text available for {filename}."

def read_text_section(basin_name: str, section: str) -> str:
    """Legacy wrapper for specific text sections."""
    return read_basin_text(basin_name, f"{section}.txt")

def find_yearly_csv(basin_name: str, year: int):
    """Find yearly CSV file for a basin and year."""
    results_dir = os.path.join(BASIN_DIR, basin_name, "Results", "yearly")
    if not os.path.isdir(results_dir):
        return None
    
    patterns = [
        os.path.join(results_dir, f"sheet1_{year}.csv"),
        os.path.join(results_dir, f"*{year}*.csv"),
        os.path.join(results_dir, "*.csv")  # Fallback to any CSV
    ]
    
    return _first_existing(patterns)

def parse_wa_sheet(csv_file: str):
    """Robust parsing of sheet1 CSV for WA+."""
    try:
        df = pd.read_csv(csv_file, sep=';')
        
        cleaned_rows = []
        for _, row in df.iterrows():
            try:
                val = float(row.get('VALUE', 0)) * 1000
            except (ValueError, TypeError):
                val = 0
            
            cleaned_rows.append({
                'CLASS': row.get('CLASS', '').strip(),
                'SUBCLASS': row.get('SUBCLASS', '').strip(),
                'VARIABLE': row.get('VARIABLE', '').strip(),
                'VALUE': val
            })
        
        return pd.DataFrame(cleaned_rows)
    except Exception as e:
        print(f"Error parsing WA sheet: {e}")
        return pd.DataFrame()

def get_wa_data(basin_name: str, start_year: int, end_year: int):
    """Aggregates WA+ data for a range of years."""
    all_data = []

    for year in range(start_year, end_year + 1):
        csv_file = find_yearly_csv(basin_name, year)
        if csv_file:
            df = parse_wa_sheet(csv_file)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by CLASS, SUBCLASS, VARIABLE and mean the VALUE
    agg_df = combined_df.groupby(['CLASS', 'SUBCLASS', 'VARIABLE'])['VALUE'].mean().reset_index()
    return agg_df

def get_basin_overview_metrics_for_range(basin_name: str, start_year: int, end_year: int):
    """Get comprehensive basin overview metrics averaged over a year range."""
    agg_df = get_wa_data(basin_name, start_year, end_year)
    if agg_df.empty:
        return None

    metrics = {}
    metrics['total_inflows'] = agg_df[agg_df['CLASS'] == 'INFLOW']['VALUE'].sum()

    metrics['total_precipitation'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION')
    ]['VALUE'].sum()

    metrics['precipitation_rainfall'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'PRECIPITATION') & (agg_df['VARIABLE'] == 'Rainfall')
    ]['VALUE'].sum()

    metrics['surface_water_imports'] = agg_df[
        (agg_df['CLASS'] == 'INFLOW') & (agg_df['SUBCLASS'] == 'SURFACE WATER') &
        (agg_df['VARIABLE'].isin(['Main riverstem', 'Tributaries']))
    ]['VALUE'].sum()

    et_rows = agg_df[(agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'].str.contains('ET'))]
    metrics['total_water_consumption'] = et_rows[~et_rows['VARIABLE'].isin(['Manmade', 'Consumed Water'])]['VALUE'].sum()

    metrics['manmade_consumption'] = agg_df[
        (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Manmade')
    ]['VALUE'].sum()

    metrics['non_irrigated_consumption'] = agg_df[
        (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'ET INCREMENTAL') & (agg_df['VARIABLE'] == 'Consumed Water')
    ]['VALUE'].sum()

    metrics['treated_wastewater'] = agg_df[
         (agg_df['CLASS'] == 'OUTFLOW') & (agg_df['SUBCLASS'] == 'OTHER') & (agg_df['VARIABLE'] == 'Treated Waste Water')
    ]['VALUE'].sum()

    recharge_val = agg_df[
        (agg_df['CLASS'] == 'STORAGE') & (agg_df['SUBCLASS'] == 'CHANGE') & (agg_df['VARIABLE'].str.contains('Surface storage'))
    ]['VALUE'].sum()
    metrics['recharge'] = abs(recharge_val) if recharge_val < 0 else recharge_val

    if metrics['total_inflows'] > 0:
        metrics['precipitation_percentage'] = (metrics['total_precipitation'] / metrics['total_inflows'] * 100)
    
    return metrics


# ======================
# INDICATOR UTILITIES
# ======================

def parse_indicators(csv_file: str):
    """Parse indicators CSV."""
    try:
        df = pd.read_csv(csv_file, sep=';')
        return df
    except Exception as e:
        print(f"Error parsing indicators: {e}")
        return pd.DataFrame()

def get_indicators(basin_name: str, start_year: int, end_year: int):
    """Aggregates indicators for a range of years."""
    all_data = []

    # We need to find indicator files
    results_dir = os.path.join(BASIN_DIR, basin_name, "Results", "indicators")
    if not os.path.isdir(results_dir):
        return pd.DataFrame()

    for year in range(start_year, end_year + 1):
        pat = os.path.join(results_dir, f"indicators_{year}.csv")
        csv_file = _first_existing([pat])
        if csv_file:
            df = parse_indicators(csv_file)
            if not df.empty:
                df['Year'] = year
                all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    numeric_cols = ['VALUE']
    meta_cols = ['UNIT', 'DEFINITION', 'TRAFFIC_LIGHT']
    
    agg_df = combined_df.groupby('INDICATOR')[numeric_cols].mean().reset_index()

    meta_df = combined_df[['INDICATOR'] + meta_cols].drop_duplicates('INDICATOR')
    agg_df = pd.merge(agg_df, meta_df, on='INDICATOR', how='left')

    return agg_df

# ======================
# VALIDATION UTILITIES
# ======================

def get_validation_data(basin_name: str, var_type: str):
    """Get validation data for rainfall or ET."""
    filename = "rainfall_validation.csv" if var_type == "P" else "et_validation.csv"
    filepath = os.path.join(BASIN_DIR, basin_name, "Results", "validation", filename)

    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, sep=';')
        return df
    except Exception as e:
        print(f"Error reading validation data: {e}")
        return pd.DataFrame()


# ===================
# SHAPEFILE UTILITIES
# ===================

def _force_2d(geom):
    try:
        return shp_wkb.loads(shp_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

def _repair_poly(geom):
    try:
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except Exception:
        return geom

def load_all_basins_geodata() -> gpd.GeoDataFrame:
    """Load ALL basins' shapefiles (exploded, fixed, EPSG:4326)."""
    rows = []
    if not os.path.isdir(BASIN_DIR):
        return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")

    for b in sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))]):
        shp = find_shp_file(b)
        if not shp or not os.path.exists(shp):
            continue
        try:
            with fiona.open(shp) as src:
                crs_wkt = src.crs_wkt
                crs_obj = None
                if crs_wkt:
                    try:
                        crs_obj = gpd.GeoSeries([0], crs=crs_wkt).crs
                    except Exception:
                        crs_obj = None

                geoms = []
                for feat in src:
                    if not feat or not feat.get("geometry"):
                        continue
                    geom = shp_shape(feat["geometry"])
                    geom = _force_2d(geom)
                    geom = _repair_poly(geom)
                    if geom and not geom.is_empty and geom.geom_type in ("Polygon", "MultiPolygon"):
                        geoms.append(geom)
                if not geoms:
                    continue

                gdf = gpd.GeoDataFrame({"basin": [b]*len(geoms)}, geometry=geoms, crs=crs_obj or "EPSG:4326")
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                except Exception:
                    gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

                try:
                    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                except Exception:
                    gdf = gdf.explode().reset_index(drop=True)

                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                rows.append(gdf[["basin", "geometry"]])
        except Exception as e:
            print(f"[WARN] Problem with {b}: {e}")
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=["basin", "geometry"], geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")

ALL_BASINS_GDF = load_all_basins_geodata()

def basins_geojson(gdf: gpd.GeoDataFrame | None = None):
    gdf = ALL_BASINS_GDF if gdf is None else gdf
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {"basin": row["basin"]},
                }
            )
        except Exception as e:
            print(f"[WARN] Could not convert geometry for basin {row['basin']}: {e}")
    return {"type": "FeatureCollection", "features": feats}


# ==============
# DATA PIPELINE
# ==============

def _compute_mode(arr, axis=None):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)] if counts.size else np.nan

def _coarsen_to_1km(da: xr.DataArray, is_categorical=False) -> xr.DataArray:
    if "latitude" not in da.dims or "longitude" not in da.dims:
        return da
    lat_vals, lon_vals = da["latitude"].values, da["longitude"].values
    lat_res = float(np.abs(np.diff(lat_vals)).mean()) if lat_vals.size > 1 else 0.009
    lon_res = float(np.abs(np.diff(lon_vals)).mean()) if lon_vals.size > 1 else 0.009
    target_deg = 1.0 / 111.0
    f_lat = max(1, int(round(target_deg / (lat_res if lat_res else target_deg))))
    f_lon = max(1, int(round(target_deg / (lon_res if lon_res else target_deg))))
    coarsen_dict = {"latitude": f_lat, "longitude": f_lon}

    if is_categorical:
        try:
            return da.coarsen(coarsen_dict, boundary="trim").reduce(_compute_mode)
        except Exception:
            return da
    else:
        try:
            return da.coarsen(coarsen_dict, boundary="trim").mean(skipna=True)
        except Exception:
            return da

def load_and_process_data(basin_name: str, variable_type: str,
                          year_start: int | None = None, year_end: int | None = None,
                          aggregate_time: bool = True):
    fp = find_nc_file(basin_name, variable_type)
    if not fp:
        return None, None, "NetCDF file not found"
    try:
        ds = _open_xr_dataset(fp)
        ds = _standardize_latlon(ds)
        var = _pick_data_var(ds)
        if not var:
            return None, None, "No suitable data variable in file"

        da = ds[var]

        if "time" in ds.coords and (year_start is not None or year_end is not None):
            ys = int(year_start) if year_start is not None else pd.to_datetime(ds["time"].values).min().year
            ye = int(year_end)   if year_end   is not None else pd.to_datetime(ds["time"].values).max().year
            da = da.sel(time=slice(f"{ys}-01-01", f"{ye}-12-31"))

        if "time" in da.dims:
            if aggregate_time and da.sizes.get("time", 0) > 1 and variable_type in ["P", "ET"]:
                da = da.mean(dim="time", skipna=True)
            elif variable_type == "LU" and da.sizes.get("time", 0) > 0:
                da = da.isel(time=-1)
            elif not aggregate_time:
                pass
            else:
                da = da.isel(time=0)

        da = _coarsen_to_1km(da, is_categorical=(variable_type == "LU"))
        return da, var, os.path.basename(fp)

    except Exception as e:
        return None, None, f"Error processing file: {e}"


# ==================
# FIGURE CONSTRUCTORS
# ==================

THEME_COLOR = "#2B587A"

def _clean_nan_data(da: xr.DataArray):
    """Remove NaN values and return clean data for plotting"""
    if da is None:
        return None, None, None
    valid_mask = np.isfinite(da.values)
    if not np.any(valid_mask):
        return None, None, None
    x = np.asarray(da["longitude"].values)
    y = np.asarray(da["latitude"].values)
    z_clean = da.values.copy()
    return z_clean, x, y

def _create_clean_heatmap(da: xr.DataArray, title: str, colorscale="Viridis", z_label="value"):
    """Create a clean heatmap that properly handles NaN values"""
    if da is None or "latitude" not in da.coords or "longitude" not in da.coords:
        return _empty_fig("No data to display")

    z, x, y = _clean_nan_data(da)
    if z is None:
        return _empty_fig("No valid data values")

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x, y=y, colorscale=colorscale, zmid=0,
        colorbar=dict(title=z_label, thickness=15, len=0.75, yanchor="middle", y=0.5),
        hoverinfo="x+y+z", hovertemplate='Longitude: %{x:.2f}<br>Latitude: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Longitude", yaxis_title="Latitude",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(color="#1e293b"), margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig

def add_shapefile_to_fig(fig: go.Figure, basin_name: str) -> go.Figure:
    """Overlay basin boundary on a cartesian image figure."""
    shp_file = find_shp_file(basin_name)
    if not shp_file or not os.path.exists(shp_file):
        return fig
    try:
        gdf = gpd.read_file(shp_file)
        try:
            gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        for geom in gdf.geometry:
            geom = _repair_poly(_force_2d(geom))
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Polygon":
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                         line=dict(color="black", width=1), showlegend=False, hoverinfo='skip'))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines",
                                             line=dict(color="black", width=1), showlegend=False, hoverinfo='skip'))
    except Exception as e:
        print(f"[WARN] Could not overlay shapefile: {e}")
    return fig

def _empty_fig(msg="No data to display"):
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": msg, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16}}],
        margin=dict(l=0, r=0, t=35, b=0), plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


# =========================
# BASIN SELECTOR (MAPBOX)
# =========================

def make_basin_selector_map(selected_basin=None) -> go.Figure:
    gdf = ALL_BASINS_GDF if (not selected_basin or selected_basin == "all") else ALL_BASINS_GDF[ALL_BASINS_GDF["basin"] == selected_basin]
    if gdf is None or gdf.empty:
        return _empty_fig("No basin shapefiles found.")

    gj = basins_geojson(gdf)
    locations = [f["properties"]["basin"] for f in gj["features"]]
    z_vals = [1] * len(locations)

    ch = go.Choroplethmapbox(
        geojson=gj, locations=locations, featureidkey="properties.basin", z=z_vals,
        colorscale=[[0, "rgba(43, 88, 122, 0.4)"], [1, "rgba(43, 88, 122, 0.4)"]], # Theme color with alpha
        marker=dict(line=dict(width=3 if selected_basin and selected_basin != "all" else 1.8, color=THEME_COLOR)),
        hovertemplate="%{location}<extra></extra>", showscale=False,
    )
    fig = go.Figure(ch)

    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.08 if maxx > minx else 0.1
    pad_y = (maxy - miny) * 0.08 if maxy > miny else 0.1
    west, east = float(minx - pad_x), float(maxx + pad_x)
    south, north = float(miny - pad_y), float(maxy + pad_y)

    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    span_lon = max(east - west, 0.001)
    span_lat = max(north - south, 0.001)

    import math
    map_w, map_h = 900.0, 600.0
    lon_zoom = math.log2(360.0 / (span_lon * 1.1)) + math.log2(map_w / 512.0)
    lat_zoom = math.log2(180.0 / (span_lat * 1.1)) + math.log2(map_h / 512.0)
    zoom = max(0.0, min(16.0, lon_zoom, lat_zoom))

    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lon=center_lon, lat=center_lat), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0), uirevision=selected_basin if selected_basin else "all", clickmode="event+select", height=400,
    )
    return fig


# Land use class information (truncated for brevity, same as before)
class_info = {
    1: {"name": "Protected forests", "color": "rgb(0,40,0)"},
    2: {"name": "Protected shrubland", "color": "rgb(190,180,60)"},
    3: {"name": "Protected natural grasslands", "color": "rgb(176,255,33)"},
    4: {"name": "Protected natural waterbodies", "color": "rgb(83,142,213)"},
    5: {"name": "Protected wetlands", "color": "rgb(40,250,180)"},
    6: {"name": "Glaciers", "color": "rgb(255,255,255)"},
    7: {"name": "Protected other", "color": "rgb(219,214,0)"},
    8: {"name": "Closed deciduous forest", "color": "rgb(0,70,0)"},
    9: {"name": "Open deciduous forest", "color": "rgb(0,124,0)"},
    10: {"name": "Closed evergreen forest", "color": "rgb(0,100,0)"},
    11: {"name": "Open evergreen forest", "color": "rgb(0,140,0)"},
    12: {"name": "Closed savanna", "color": "rgb(155,150,50)"},
    13: {"name": "Open savanna", "color": "rgb(255,190,90)"},
    14: {"name": "Shrub land & mesquite", "color": "rgb(120,150,30)"},
    15: {"name": "Herbaceous cover", "color": "rgb(90,115,25)"},
    16: {"name": "Meadows & open grassland", "color": "rgb(140,190,100)"},
    17: {"name": "Riparian corridor", "color": "rgb(30,190,170)"},
    18: {"name": "Deserts", "color": "rgb(245,255,230)"},
    19: {"name": "Wadis", "color": "rgb(200,230,255)"},
    20: {"name": "Natural alpine pastures", "color": "rgb(86,134,0)"},
    21: {"name": "Rocks & gravel & stones & boulders", "color": "rgb(255,210,110)"},
    22: {"name": "Permafrosts", "color": "rgb(230,230,230)"},
    23: {"name": "Brooks & rivers & waterfalls", "color": "rgb(0,100,240)"},
    24: {"name": "Natural lakes", "color": "rgb(0,55,154)"},
    25: {"name": "Flood plains & mudflats", "color": "rgb(165,230,100)"},
    26: {"name": "Saline sinks & playas & salinized soil", "color": "rgb(210,230,210)"},
    27: {"name": "Bare soil", "color": "rgb(240,165,20)"},
    28: {"name": "Waste land", "color": "rgb(230,220,210)"},
    29: {"name": "Moorland", "color": "rgb(190,160,140)"},
    30: {"name": "Wetland", "color": "rgb(33,193,132)"},
    31: {"name": "Mangroves", "color": "rgb(28,164,112)"},
    32: {"name": "Alien invasive species", "color": "rgb(100,255,150)"},
    33: {"name": "Rainfed forest plantations", "color": "rgb(245,250,194)"},
    34: {"name": "Rainfed production pastures", "color": "rgb(237,246,152)"},
    35: {"name": "Rainfed crops - cereals", "color": "rgb(226,240,90)"},
    36: {"name": "Rainfed crops - root/tuber", "color": "rgb(209,229,21)"},
    37: {"name": "Rainfed crops - legumious", "color": "rgb(182,199,19)"},
    38: {"name": "Rainfed crops - sugar", "color": "rgb(151,165,15)"},
    39: {"name": "Rainfed crops - fruit and nuts", "color": "rgb(132,144,14)"},
    40: {"name": "Rainfed crops - vegetables and melons", "color": "rgb(112,122,12)"},
    41: {"name": "Rainfed crops - oilseed", "color": "rgb(92,101,11)"},
    42: {"name": "Rainfed crops - beverage and spice", "color": "rgb(71,80,8)"},
    43: {"name": "Rainfed crops - other", "color": "rgb(51,57,5)"},
    44: {"name": "Mixed species agro-forestry", "color": "rgb(80,190,40)"},
    45: {"name": "Fallow & idle land", "color": "rgb(180,160,180)"},
    46: {"name": "Dump sites & deposits", "color": "rgb(145,130,115)"},
    47: {"name": "Rainfed homesteads and gardens (urban cities) - outdoor", "color": "rgb(120,5,25)"},
    48: {"name": "Rainfed homesteads and gardens (rural villages) - outdoor", "color": "rgb(210,10,40)"},
    49: {"name": "Rainfed industry parks - outdoor", "color": "rgb(255,130,45)"},
    50: {"name": "Rainfed parks (leisure & sports)", "color": "rgb(250,101,0)"},
    51: {"name": "Rural paved surfaces (lots, roads, lanes)", "color": "rgb(255,150,150)"},
    52: {"name": "Irrigated forest plantations", "color": "rgb(179,243,241)"},
    53: {"name": "Irrigated production pastures", "color": "rgb(158,240,238)"},
    54: {"name": "Irrigated crops - cereals", "color": "rgb(113,233,230)"},
    55: {"name": "Irrigated crops - root/tubers", "color": "rgb(82,228,225)"},
    56: {"name": "Irrigated crops - legumious", "color": "rgb(53,223,219)"},
    57: {"name": "Irrigated crops - sugar", "color": "rgb(33,205,201)"},
    58: {"name": "Irrigated crops - fruit and nuts", "color": "rgb(29,179,175)"},
    59: {"name": "Irrigated crops - vegetables and melons", "color": "rgb(25,151,148)"},
    60: {"name": "Irrigated crops - Oilseed", "color": "rgb(21,125,123)"},
    61: {"name": "Irrigated crops - beverage and spice", "color": "rgb(17,101,99)"},
    62: {"name": "Irrigated crops - other", "color": "rgb(13,75,74)"},
    63: {"name": "Managed water bodies (reservoirs, canals, harbors, tanks)", "color": "rgb(0,40,112)"},
    64: {"name": "Greenhouses - indoor", "color": "rgb(255,204,255)"},
    65: {"name": "Aquaculture", "color": "rgb(47,121,255)"},
    66: {"name": "Domestic households - indoor (sanitation)", "color": "rgb(255,60,10)"},
    67: {"name": "Manufacturing & commercial industry - indoor", "color": "rgb(180,180,180)"},
    68: {"name": "Irrigated homesteads and gardens (urban cities) - outdoor", "color": "rgb(255,139,255)"},
    69: {"name": "Irrigated homesteads and gardens (rural villages) - outdoor", "color": "rgb(255,75,255)"},
    70: {"name": "Irrigated industry parks - outdoor", "color": "rgb(140,140,140)"},
    71: {"name": "Irrigated parks (leisure, sports)", "color": "rgb(150,0,205)"},
    72: {"name": "Urban paved Surface (lots, roads, lanes)", "color": "rgb(120,120,120)"},
    73: {"name": "Livestock and domestic husbandry", "color": "rgb(180,130,130)"},
    74: {"name": "Managed wetlands & swamps", "color": "rgb(30,130,115)"},
    75: {"name": "Managed other inundation areas", "color": "rgb(20,150,130)"},
    76: {"name": "Mining/ quarry & shale exploiration", "color": "rgb(100,100,100)"},
    77: {"name": "Evaporation ponds", "color": "rgb(30,90,130)"},
    78: {"name": "Waste water treatment plants", "color": "rgb(60,60,60)"},
    79: {"name": "Hydropower plants", "color": "rgb(40,40,40)"},
    80: {"name": "Thermal power plants", "color": "rgb(0,0,0)"},
}

# ===========
# DASH APP
# ===========

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Water Accounting Jordan"

basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{"label": "Select a Basin...", "value": "none"}] + [{"label": b, "value": b} for b in sorted(basin_folders)]

# Static Text Content
INTRO_TEXT = read_common_text("intro.txt")
OBJECTIVES_TEXT = read_common_text("objectives.txt")

WA_FRAMEWORK_TEXT = """
WA+ is a robust framework that harnesses the potential of publicly available remote sensing data to assess water resources and their consumption. Its reliance on such data is particularly beneficial in data scarce areas and transboundary basins. A significant benefit of WA+ lies in its incorporation of land use classification into water resource assessments, promoting a holistic approach to land and water management. This integration is crucial for sustaining food production amidst a changing climate, especially in regions where water is scarce. Notably, WA+ application has predominantly centered on monitoring water consumption in irrigated agriculture. The WA+ approach builds on a simplified water balance equation for a basin (Karimi et al., 2013):

**∆S/∆t = P - ET - Q_out**                                                                                   (1)

Where:
*   **∆S** is the change in storage
*   **∆t** is the change in time
*   **P** is precipitation (mm/year or m3/year)
*   **ET** is total actual evapotranspiration (mm/year or m3/year)
*   **Qout** is total surface water outflow (mm/year or m3/year)

To utilize the WA+ approach for water budget reporting in Jordan, it is important to account for all water users, other than irrigation, and their return flows into equation 1. Also, in Jordan, man-made inflows and outflows of great importance especially in heavily populated basins (Amdar et al., 2024). Therefore, an updated water balance incorporating various sectoral water consumption in addition to inflow and outflows is proposed (Amdar et al., 2024). Hence, equation (2) represents the updated WA+ water balance equation in the context of Jordan. This modification will further be refined following detailed discussions and consultations with the WEC and MWI team to ensure complete understanding and consensus of the customized framework for Jordan.

**∆S/∆t = (P + Q_in) - (ET + CW_sec + Q_WWT + Q_re + Q_natural)**                               (2)

where:
*   **P** is the total precipitation (Mm3/year)
*   **ET** is the total actual evapotranspiration (Mm3/year)
*   **Qin** is the total inflows into the basin consisting of both surface water inflows and any other inter-basin transfers (Mm3/year)
*   **Qre** is the total recharge to groundwater from precipitation and return flow (Mm3/year)
*   **QWWT** is the total treated waste water that is returned to the river system after treatment. This could be from domestic, industry and tourism sectors (Mm3/year)
*   **Qnatural** is the naturalized streamflow from the basin (Mm3/year)
*   **CWsec** is the total non-irrigated water use/consumption (ie water that is not returned to the system but is consumed by humans) and is given by:

**CWsec = Supplydomestic + Supplyindustrial + Supplylivestock + Supplytourism**
(3)

Where:
*   **Supplydomestic** is the water supply for the domestic sector (Mm3/year)
*   **Supplyindustrial** is the water supply for the industrial sector (Mm3/year)
*   **Supplylivestock** is the water supply for the livestock sector (Mm3/year)
*   **Supplytourism** is the water supply for the tourism sector (Mm3/year)

The customized WA+ framework thus takes into account both agricultural and non-irrigated water consumption, water imports and the return of treated wastewater into the basin.
"""

LAND_USE_DATA = [
    {"Class": "Natural", "Subclass": "Protected forests", "Area_Sub_km2": 6.14, "Area_Class_km2": 3378.9, "Area_Pct": 70, "P": 552.2, "ET": 633.8, "P_ET": -81.6},
    {"Class": "Natural", "Subclass": "Protected shrubland", "Area_Sub_km2": 4.15, "Area_Class_km2": None, "Area_Pct": None, "P": 522.4, "ET": 583.1, "P_ET": -60.7},
    {"Class": "Natural", "Subclass": "Protected other", "Area_Sub_km2": 26.38, "Area_Class_km2": None, "Area_Pct": None, "P": 416.8, "ET": 394.0, "P_ET": 22.8},
    {"Class": "Natural", "Subclass": "Open deciduous forest", "Area_Sub_km2": 26.93, "Area_Class_km2": None, "Area_Pct": None, "P": 412.3, "ET": 452.6, "P_ET": -40.3},
    {"Class": "Natural", "Subclass": "Closed evergreen forest", "Area_Sub_km2": 0.62, "Area_Class_km2": None, "Area_Pct": None, "P": 583.6, "ET": 727.3, "P_ET": -143.6},
    {"Class": "Natural", "Subclass": "Shrub land & mesquite", "Area_Sub_km2": 211.79, "Area_Class_km2": None, "Area_Pct": None, "P": 407.2, "ET": 411.8, "P_ET": -4.6},
    {"Class": "Natural", "Subclass": "Meadows & open grassland", "Area_Sub_km2": 1290.45, "Area_Class_km2": None, "Area_Pct": None, "P": 284.9, "ET": 174.3, "P_ET": 110.6},
    {"Class": "Natural", "Subclass": "Fallow & idle land", "Area_Sub_km2": 1812.49, "Area_Class_km2": None, "Area_Pct": None, "P": 178.3, "ET": 24.4, "P_ET": 153.9},
    {"Class": "Agricultural", "Subclass": "Rainfed crops", "Area_Sub_km2": 208.90, "Area_Class_km2": 1105.0, "Area_Pct": 23, "P": 285.7, "ET": 235.9, "P_ET": 49.8},
    {"Class": "Agricultural", "Subclass": "Rainfed crops - other", "Area_Sub_km2": 818.83, "Area_Class_km2": None, "Area_Pct": None, "P": 209.8, "ET": 73.0, "P_ET": 136.8},
    {"Class": "Agricultural", "Subclass": "Irrigated crops", "Area_Sub_km2": 75.68, "Area_Class_km2": None, "Area_Pct": None, "P": 202.6, "ET": 334.0, "P_ET": -131.4},
    {"Class": "Agricultural", "Subclass": "Managed water bodies", "Area_Sub_km2": 1.56, "Area_Class_km2": None, "Area_Pct": None, "P": 538.9, "ET": 1045.7, "P_ET": -506.9},
    {"Class": "Urban", "Subclass": "Urban paved Surface", "Area_Sub_km2": 345.97, "Area_Class_km2": 346.0, "Area_Pct": 7, "P": 268.2, "ET": 138.0, "P_ET": 130.1},
    {"Class": "Total", "Subclass": "", "Area_Sub_km2": 4829.89, "Area_Class_km2": 4829.89, "Area_Pct": 100, "P": None, "ET": None, "P_ET": None},
]


# ==================
# LAYOUT COMPONENTS
# ==================

def get_header():
    return html.Nav(
        className="navbar-custom",
        style={"backgroundColor": THEME_COLOR, "padding": "15px 20px", "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
        children=[
            html.Div(className="navbar-brand-group", style={"display": "flex", "alignItems": "center"}, children=[
                html.Img(src=app.get_asset_url('iwmi.png'), style={"height": "50px", "marginRight": "15px", "filter": "brightness(0) invert(1)"}),
                html.H1("Water Accounting Jordan", style={"color": "white", "margin": 0, "fontSize": "1.5rem", "fontWeight": "600", "fontFamily": "Segoe UI, sans-serif"}),
            ]),
            html.Div(className="nav-links", style={"display": "flex", "alignItems": "center"}, children=[
                html.Img(src=app.get_asset_url('cgiar.png'), style={"height": "40px", "filter": "brightness(0) invert(1)"}),
            ])
        ]
    )

def get_footer():
    return html.Footer(className="site-footer", style={"backgroundColor": THEME_COLOR, "color": "white", "padding": "40px 20px", "marginTop": "40px"}, children=[
        html.Div(className="footer-content", style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap", "maxWidth": "1200px", "margin": "0 auto"}, children=[
            html.Div(className="footer-col", style={"flex": "1", "minWidth": "250px", "marginBottom": "20px"}, children=[
                html.H4("International Water Management Institute", style={"fontSize": "1.1rem", "fontWeight": "bold", "marginBottom": "15px"}),
                html.P("Science for a water-secure world.", style={"color": "rgba(255,255,255,0.7)", "fontSize": "0.9rem"})
            ]),
            html.Div(className="footer-col", style={"flex": "1", "minWidth": "250px", "marginBottom": "20px"}, children=[
                html.H4("Contact", style={"fontSize": "1.1rem", "fontWeight": "bold", "marginBottom": "15px"}),
                html.P("127 Sunil Mawatha, Pelawatte, Battaramulla, Sri Lanka", style={"color": "rgba(255,255,255,0.7)", "fontSize": "0.9rem", "marginBottom": "5px"}),
                html.P("iwmi@cgiar.org", style={"color": "rgba(255,255,255,0.7)", "fontSize": "0.9rem"})
            ])
        ]),
        html.Div(className="footer-bottom", style={"textAlign": "center", "borderTop": "1px solid rgba(255,255,255,0.1)", "paddingTop": "20px", "marginTop": "20px"}, children=[
            html.P("© 2024 International Water Management Institute (IWMI). All rights reserved.", style={"fontSize": "0.85rem", "color": "rgba(255,255,255,0.6)"})
        ])
    ])

def get_home_content():
    return html.Div([
        # Hero Section
        html.Div(className="hero-section", style={
            "backgroundImage": "linear-gradient(rgba(43, 88, 122, 0.7), rgba(43, 88, 122, 0.8)), url('https://images.unsplash.com/photo-1589923188900-85dae5233271?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80')",
            "backgroundSize": "cover",
            "backgroundPosition": "center",
            "padding": "100px 20px",
            "textAlign": "center",
            "color": "white",
            "marginBottom": "40px"
        }, children=[
            html.H1("Rapid Water Accounting Dashboard - Jordan", style={"fontSize": "3.5rem", "fontWeight": "700", "marginBottom": "1rem"}),
            html.P("Empowering sustainable water management through advanced remote sensing data and hydrological modeling.", style={"fontSize": "1.5rem", "fontWeight": "300", "maxWidth": "800px", "margin": "0 auto"}),
        ]),
        # Features Section
        html.Div(className="content-section", style={"maxWidth": "1200px", "margin": "0 auto", "padding": "0 20px"}, children=[
            html.Div(className="grid-3", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))", "gap": "30px"}, children=[
                html.Div(className="feature-card", style={"backgroundColor": "white", "padding": "30px", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderTop": f"5px solid {THEME_COLOR}"}, children=[
                    html.Div("📊", style={"fontSize": "3rem", "marginBottom": "15px"}),
                    html.H3("Basin Analysis", style={"color": THEME_COLOR, "fontWeight": "600", "marginBottom": "10px"}),
                    html.P("Interactive maps and metrics for major basins in Jordan. Analyze inflows, outflows, and storage changes.", style={"color": "#666", "lineHeight": "1.6"})
                ]),
                html.Div(className="feature-card", style={"backgroundColor": "white", "padding": "30px", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderTop": f"5px solid {THEME_COLOR}"}, children=[
                    html.Div("🌧️", style={"fontSize": "3rem", "marginBottom": "15px"}),
                    html.H3("Climate Data", style={"color": THEME_COLOR, "fontWeight": "600", "marginBottom": "10px"}),
                    html.P("Visualize long-term precipitation and evapotranspiration trends derived from high-resolution satellite data.", style={"color": "#666", "lineHeight": "1.6"})
                ]),
                html.Div(className="feature-card", style={"backgroundColor": "white", "padding": "30px", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "borderTop": f"5px solid {THEME_COLOR}"}, children=[
                    html.Div("📑", style={"fontSize": "3rem", "marginBottom": "15px"}),
                    html.H3("WA+ Reporting", style={"color": THEME_COLOR, "fontWeight": "600", "marginBottom": "10px"}),
                    html.P("Standardized Water Accounting Plus (WA+) sheets and indicators to support evidence-based decision making.", style={"color": "#666", "lineHeight": "1.6"})
                ])
            ])
        ]),
    ])

def get_land_use_layout(basin):
    """Generates the Land Use section (Static / Last Year)."""
    # Read Land Use Text
    lu_text = read_basin_text(basin, "lu.txt")

    # Table logic
    table_component = html.Div("No table data available.")
    if basin == "Amman Zarqa":
        df = pd.DataFrame(LAND_USE_DATA)
        table_component = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[
                {'name': 'Water Management Class', 'id': 'Class'},
                {'name': 'Land and water use', 'id': 'Subclass'},
                {'name': 'Area (km2)', 'id': 'Area_Sub_km2'},
                {'name': 'Area (km2)', 'id': 'Area_Class_km2'},
                {'name': 'Area (%)', 'id': 'Area_Pct'},
                {'name': 'P (mm)', 'id': 'P'},
                {'name': 'ET (mm)', 'id': 'ET'},
                {'name': 'P-ET (mm)', 'id': 'P_ET'},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '12px',
                'fontFamily': 'Segoe UI, sans-serif',
                'border': '1px solid #e2e8f0'
            },
            style_header={
                'backgroundColor': '#eff6ff', # Light blue
                'fontWeight': 'bold',
                'color': THEME_COLOR,
                'border': '1px solid #e2e8f0'
            },
            style_data_conditional=[
                 {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8fafc'}
            ]
        )

    return html.Div([
        html.H3("Land Use (Latest Year)", style={"color": THEME_COLOR, "marginTop": "20px", "borderBottom": f"2px solid {THEME_COLOR}", "paddingBottom": "10px"}),

        dcc.Markdown(lu_text, className="markdown-content"),
        html.Div(table_component, style={"marginTop": "20px", "marginBottom": "30px", "overflowX": "auto"}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id="lu-map-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
            html.Div(dcc.Loading(dcc.Graph(id="lu-bar-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "float": "right", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
        ]),
    ], id="section-land-use")

def get_climate_inputs_layout(basin):
    """Generates Precipitation, ET, and Validation sections."""
    return html.Div([
        # Precipitation Subsection
        html.H4("Precipitation", style={"color": THEME_COLOR, "marginTop": "40px", "fontSize": "1.5rem"}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id="p-map-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
            html.Div(dcc.Loading(dcc.Graph(id="p-bar-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "float": "right", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
        ]),
        html.Div(id="p-explanation", className="graph-card", style={"marginTop":"10px"}),

        # ET Subsection
        html.H4("Evapotranspiration", style={"color": THEME_COLOR, "marginTop": "40px", "fontSize": "1.5rem"}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id="et-map-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
            html.Div(dcc.Loading(dcc.Graph(id="et-bar-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "float": "right", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
        ]),
        html.Div(id="et-explanation", className="graph-card", style={"marginTop":"10px"}),

        # Validation
        html.H4("Data Validation", style={"color": THEME_COLOR, "marginTop": "40px", "fontSize": "1.5rem"}),
         html.Div([
             html.Div(dcc.Graph(id="val-p-scatter"), style={"width": "49%", "display": "inline-block", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
             html.Div(dcc.Graph(id="val-et-scatter"), style={"width": "49%", "display": "inline-block", "float": "right", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"})
        ])
    ], id="section-climate-inputs")


def get_results_layout(basin):
    return html.Div([
        html.H3("Results", style={"color": THEME_COLOR, "marginTop": "40px", "borderBottom": f"2px solid {THEME_COLOR}", "paddingBottom": "10px"}),

        # Overview
        html.H4("Basin Overview", style={"color": THEME_COLOR, "fontSize": "1.5rem"}),
        html.Div(id="basin-overview-content"),

        # Water Balance
        html.H4("Water Balance (P - ET)", style={"color": THEME_COLOR, "marginTop": "30px", "fontSize": "1.5rem"}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id="p-et-map-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
            html.Div(dcc.Loading(dcc.Graph(id="p-et-bar-graph"), type="circle"), style={"width": "49%", "display": "inline-block", "float": "right", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "borderRadius": "8px"}),
        ]),
        html.Div(id="p-et-explanation", className="graph-card", style={"marginTop":"10px"}),

        # WA+ Sheets
        html.H4("Water Accounting Reports", style={"color": THEME_COLOR, "marginTop": "30px", "fontSize": "1.5rem"}),
        dcc.Loading(dcc.Graph(id="wa-resource-base-sankey"), type="circle"),
        dcc.Loading(dcc.Graph(id="wa-sectoral-bar"), type="circle"),
        html.Div(id="wa-indicators-container"), # Added container for indicators

        # Reports Tabs
        html.H4("Documentation", style={"color": THEME_COLOR, "marginTop": "40px", "fontSize": "1.5rem"}),
        html.Div([
            dcc.Tabs(id="inner-report-tabs", value="assumptions", vertical=True, children=[
                dcc.Tab(label="Assumptions", value="assumptions", style={'color': THEME_COLOR}, selected_style={'fontWeight': 'bold', 'color': THEME_COLOR, 'borderLeft': f'3px solid {THEME_COLOR}'}),
                dcc.Tab(label="Limitations", value="limitations", style={'color': THEME_COLOR}, selected_style={'fontWeight': 'bold', 'color': THEME_COLOR, 'borderLeft': f'3px solid {THEME_COLOR}'}),
            ], style={"height": "300px", "width": "20%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div(id="report-content", style={"width": "75%", "display": "inline-block", "marginLeft": "2%", "padding": "25px", "backgroundColor": "white", "borderRadius": "8px", "boxShadow": "0 2px 8px rgba(0,0,0,0.05)", "border": "1px solid #eee"})
        ])
    ], id="section-results")

# Define the app layout
app.layout = html.Div([
    get_header(),
    dbc.Tabs(id="main-tabs", active_tab="tab-home", style={"marginTop": "20px", "marginLeft": "20px", "marginRight": "20px"}, children=[
        dbc.Tab(label="Home", tab_id="tab-home", label_style={"color": THEME_COLOR, "fontWeight": "600"}),
        dbc.Tab(label="Introduction", tab_id="tab-intro", label_style={"color": THEME_COLOR, "fontWeight": "600"}),
        dbc.Tab(label="Framework", tab_id="tab-framework", label_style={"color": THEME_COLOR, "fontWeight": "600"}),
        dbc.Tab(label="WA+ Analysis", tab_id="tab-analysis", label_style={"color": THEME_COLOR, "fontWeight": "600"}),
    ]),
    html.Div(id="tab-content", style={"padding": "20px", "minHeight": "600px", "backgroundColor": "#F8F9FA"}),
    get_footer()
])


# === CALLBACKS ===

@app.callback(Output("tab-content", "children"), [Input("main-tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-home":
        return get_home_content()

    elif active_tab == "tab-intro":
        return html.Div(className="container", style={"maxWidth": "1200px"}, children=[
             html.Div(className="graph-card", style={"padding": "30px", "backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "marginBottom": "30px"}, children=[
                html.H2("Introduction", style={"color": THEME_COLOR, "marginBottom": "20px"}),
                dcc.Markdown(INTRO_TEXT, className="markdown-content")
            ]),
            html.Div(className="graph-card", style={"padding": "30px", "backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"}, children=[
                html.H2("Objectives and Deliverables", style={"color": THEME_COLOR, "marginBottom": "20px"}),
                dcc.Markdown(OBJECTIVES_TEXT, className="markdown-content")
            ])
        ])

    elif active_tab == "tab-framework":
        return html.Div(className="container", style={"maxWidth": "1200px"}, children=[
            html.Div(className="graph-card", style={"padding": "30px", "backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"}, children=[
                html.H2("Customized WA+ Analytics for Jordan", style={"color": THEME_COLOR, "marginBottom": "20px"}),
                dcc.Markdown(WA_FRAMEWORK_TEXT, className="markdown-content")
            ])
        ])

    elif active_tab == "tab-analysis":
        return html.Div(className="container", style={"maxWidth": "1200px"}, children=[
            html.Div(
                className="filters-panel",
                style={"marginTop": "20px"},
                children=[
                    html.Div(children=[
                        html.H3("Select Basin", style={"color": THEME_COLOR, "marginBottom": "20px", "fontWeight": "600", "fontSize": "1.8rem"}),
                        html.Div([
                            html.Div([
                                    html.Label("Choose from list:", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block", "color": THEME_COLOR}),
                                    dcc.Dropdown(
                                    id="basin-dropdown",
                                    options=basin_options,
                                    value=None,
                                    placeholder="Select a basin...",
                                    style={"borderRadius": "4px"},
                                    persistence=True,
                                    persistence_type="session"
                                ),
                                # Study Area Text Area
                                html.Div(id="study-area-container", style={"marginTop": "20px", "padding": "20px", "backgroundColor": "#f0f4f8", "borderRadius": "8px", "fontSize": "1rem", "lineHeight": "1.8", "color": "#2c3e50", "textAlign": "justify", "borderLeft": f"4px solid {THEME_COLOR}"})
                            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),

                            html.Div([
                                    dcc.Graph(id="basin-map", style={"height": "400px", "borderRadius": "8px", "overflow": "hidden"})
                            ], style={"width": "68%", "display": "inline-block", "marginLeft": "2%", "verticalAlign": "top", "boxShadow": "0 4px 12px rgba(0,0,0,0.1)", "borderRadius": "8px"})
                        ])
                    ])
                ]
            ),
            html.Div(id="dynamic-content")
        ])

    return html.Div("404")

@app.callback(
    Output("basin-map", "figure"),
    [Input("basin-dropdown", "value")]
)
def update_map(basin):
    if basin == "none": basin = None
    return make_basin_selector_map(selected_basin=basin)

@app.callback(
    Output("basin-dropdown", "value"),
    [Input("basin-map", "clickData")],
    [State("basin-dropdown", "value")]
)
def map_click(clickData, current):
    if clickData and "points" in clickData:
        return clickData["points"][0].get("location", current)
    return current

@app.callback(
    Output("study-area-container", "children"),
    [Input("basin-dropdown", "value")]
)
def update_study_area_text(basin):
    if not basin or basin == "none" or basin == "all":
        return "Select a basin to view study area details."

    text = read_basin_text(basin, "study area.txt")
    if "No text available" in text:
        text = read_basin_text(basin, "studyarea.txt") # Fallback to no-space version if needed

    return [html.H4(f"{basin} Study Area", style={"marginTop": "0", "color": THEME_COLOR}), dcc.Markdown(text)]

def get_year_options(basin):
    p_fp = find_nc_file(basin, "P")
    et_fp = find_nc_file(basin, "ET")

    p_min, p_max = 2000, 2020
    et_min, et_max = 2000, 2020

    try:
        if p_fp:
            with _open_xr_dataset(p_fp) as ds:
                if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                    t = pd.to_datetime(ds["time"].values)
                    p_min, p_max = int(t.min().year), int(t.max().year)
        if et_fp:
             with _open_xr_dataset(et_fp) as ds:
                if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
                    t = pd.to_datetime(ds["time"].values)
                    et_min, et_max = int(t.min().year), int(t.max().year)
    except:
        pass

    start = min(p_min, et_min)
    end = max(p_max, et_max)

    if start > end:
        start, end = 2000, 2020

    years = list(range(start, end + 1))
    opts = [{"label": str(y), "value": y} for y in years]
    return opts, start, end


@app.callback(
    Output("dynamic-content", "children"),
    [Input("basin-dropdown", "value")]
)
def render_basin_content(basin):
    if not basin or basin == "none" or basin == "all":
        return html.Div(
            style={"padding": "80px", "textAlign": "center", "color": "#64748b"},
            children=[html.H3("Please select a basin above to view the analysis.", style={"color": THEME_COLOR})]
        )

    # Get years
    opts, start, end = get_year_options(basin)
    default_start = start
    default_end = end

    content = []

    # 1. Land Use Section (Fixed/Latest Year - No Year Selection dependence)
    content.append(html.Div(className="graph-card", style={"padding": "30px", "backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "marginTop": "30px"}, children=[
        get_land_use_layout(basin)
    ]))

    # 2. Climate & Results with Year Selection
    content.append(html.Div(className="graph-card", style={"padding": "30px", "backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)", "marginTop": "30px"}, children=[

        # Global Settings for Climate/Results
        html.Div(style={"backgroundColor": "#f8fafc", "padding": "25px", "borderRadius": "8px", "marginBottom": "40px", "borderLeft": f"5px solid {THEME_COLOR}"}, children=[
            html.H4("Analysis Settings (For Climate & Results)", style={"marginTop": "0", "color": THEME_COLOR, "marginBottom": "15px"}),
            html.Div([
                html.Div([
                    html.Label("Start Year", style={"fontWeight": "bold", "color": "#2c3e50"}),
                    dcc.Dropdown(id="global-start-year-dropdown", options=opts, value=default_start, clearable=False, style={"backgroundColor": "white"})
                ], style={"width": "200px", "display": "inline-block", "marginRight": "30px"}),
                html.Div([
                    html.Label("End Year", style={"fontWeight": "bold", "color": "#2c3e50"}),
                    dcc.Dropdown(id="global-end-year-dropdown", options=opts, value=default_end, clearable=False, style={"backgroundColor": "white"})
                ], style={"width": "200px", "display": "inline-block"})
            ])
        ]),

        # Sections that depend on Year Selection
        get_climate_inputs_layout(basin),
        get_results_layout(basin)
    ]))

    return content

@app.callback(
    Output("report-content", "children"),
    [Input("inner-report-tabs", "value"), Input("basin-dropdown", "value")]
)
def update_inner_report(tab, basin):
    if not basin or basin == "none": return ""
    return dcc.Markdown(read_text_section(basin, tab), className="markdown-content")

# --- DATA PROCESSING LOGIC & WRAPPERS ---

def update_basin_overview(basin, start_year, end_year):
    if not basin or basin == "none" or not start_year or not end_year:
        return html.Div("Select a specific basin and year range to view overview metrics.", 
                       style={"textAlign": "center", "color": "#64748b", "padding": "40px"})
    
    try:
        start_year, end_year = int(start_year), int(end_year)
        metrics = get_basin_overview_metrics_for_range(basin, start_year, end_year)
        
        if not metrics:
            return html.Div(f"No overview data available for {basin} in {start_year}-{end_year}.")
        
        total_inflows = f"{metrics.get('total_inflows', 0):.0f}"
        precip_pct = f"{metrics.get('precipitation_percentage', 0):.0f}"
        imports = f"{metrics.get('surface_water_imports', 0):.0f}"
        total_consumption = f"{metrics.get('total_water_consumption', 0):.0f}"
        manmade_consumption = f"{metrics.get('manmade_consumption', 0):.0f}"
        treated_wastewater = f"{metrics.get('treated_wastewater', 0):.0f}"
        non_irrigated = f"{metrics.get('non_irrigated_consumption', 0):.0f}"
        recharge = f"{metrics.get('recharge', 0):.0f}"
        
        summary_items = [
            f"Total water inflows: {total_inflows} Mm3/year.",
            f"Precipitation is {precip_pct}% of gross inflows.",
            f"Imported water: {imports} Mm3/year.",
            f"Total landscape consumption: {total_consumption} Mm3/year.",
            f"Manmade consumption: {manmade_consumption} Mm3/year",
            f"Treated wastewater discharged: {treated_wastewater} Mm3/year.",
            f"Non-irrigated consumption: {non_irrigated} Mm3/year.",
            f"Groundwater recharge: {recharge} Mm3/year."
        ]
        
        key_metrics = [
            {'title': 'Total Inflows', 'value': metrics.get('total_inflows', 0), 'unit': 'Mm3', 'color': '#3b82f6'},
            {'title': 'Precipitation', 'value': metrics.get('total_precipitation', 0), 'unit': 'Mm3', 'color': '#06b6d4'},
            {'title': 'Consumption', 'value': metrics.get('total_water_consumption', 0), 'unit': 'Mm3', 'color': '#ef4444'},
            {'title': 'Recharge', 'value': metrics.get('recharge', 0), 'unit': 'Mm3', 'color': '#10b981'}
        ]
        
        metric_cards = []
        for m in key_metrics:
            metric_cards.append(html.Div([
                html.H4(m['title'], style={"fontSize": "14px", "color": "#64748b", "marginBottom": "5px"}),
                html.Div(f"{m['value']:.0f} {m['unit']}", style={"fontSize": "24px", "fontWeight": "bold", "color": m['color']})
            ], style={"display": "inline-block", "width": "23%", "margin": "1%", "padding": "20px", "backgroundColor": "white", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}))

        return html.Div([
            html.Div(metric_cards, style={"marginBottom": "20px"}),
            html.Div([
                html.H5("Executive Summary", style={"color": THEME_COLOR, "fontWeight": "bold", "marginBottom": "10px"}),
                html.Ul([html.Li(item, style={"marginBottom": "8px"}) for item in summary_items], style={"paddingLeft": "20px"})
            ], style={"padding": "20px", "backgroundColor": "#eff6ff", "borderRadius": "8px", "borderLeft": f"4px solid {THEME_COLOR}", "color": "#2c3e50"})
        ])
    except Exception as e:
        return html.Div(f"Error: {e}")

def _generate_explanation(vtype: str, basin: str, start_year: int, end_year: int, y_vals: np.ndarray, months: list):
    mean_val = np.nanmean(y_vals)
    max_val = np.nanmax(y_vals)
    min_val = np.nanmin(y_vals)
    max_month = months[np.nanargmax(y_vals)]
    min_month = months[np.nanargmin(y_vals)]
    
    if vtype == "P":
        return (f"**Precipitation ({start_year}–{end_year}):** Average monthly P is **{mean_val:.2f} mm**. "
                f"Peak in **{max_month}** (**{max_val:.2f} mm**), lowest in **{min_month}**.")
    elif vtype == "ET":
        return (f"**Evapotranspiration ({start_year}–{end_year}):** Average monthly ET is **{mean_val:.2f} mm**. "
                f"Peak in **{max_month}** (**{max_val:.2f} mm**).")
    elif vtype == "P-ET":
        return (f"**Water Balance ({start_year}–{end_year}):** Average monthly P-ET is **{mean_val:.2f} mm**. "
                f"Max surplus in **{max_month}**, max deficit in **{min_month}**.")
    return ""

def _hydro_figs(basin: str, start_year: int | None, end_year: int | None, vtype: str):
    if not basin or basin == "none": return _empty_fig(), _empty_fig(), ""
    if not start_year or not end_year: return _empty_fig(), _empty_fig(), ""

    ys, ye = int(start_year), int(end_year)
    da_ts, _, msg = load_and_process_data(basin, vtype, year_start=ys, year_end=ye, aggregate_time=False)

    if da_ts is None: return _empty_fig(msg), _empty_fig(), msg

    da_map = da_ts.mean(dim="time", skipna=True)
    colorscale = "Blues" if vtype == "P" else "YlOrRd"
    fig_map = _create_clean_heatmap(da_map, f"Mean {vtype}", colorscale, "mm")
    fig_map = add_shapefile_to_fig(fig_map, basin)

    spatial_mean_ts = da_ts.mean(dim=["latitude", "longitude"], skipna=True)
    try:
        monthly = spatial_mean_ts.groupby("time.month").mean(skipna=True).rename({"month": "Month"})
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly["Month"].values]
        y_vals = np.asarray(monthly.values).flatten()
        fig_bar = px.bar(x=months, y=y_vals, title=f"Mean Monthly {vtype}")
        fig_bar.update_traces(marker_color=THEME_COLOR)
        fig_bar.update_layout(plot_bgcolor='white', font=dict(family="Segoe UI"))
        explanation = _generate_explanation(vtype, basin, ys, ye, y_vals, months)
    except:
        fig_bar = _empty_fig("Data Error")
        explanation = "Error"
        
    return fig_map, fig_bar, dcc.Markdown(explanation, className="markdown-content")

def update_p_et_outputs(basin, start_year, end_year):
    if not basin or basin == "none" or not start_year or not end_year:
         return _empty_fig(), _empty_fig(), ""

    ys, ye = int(start_year), int(end_year)
    da_p, _, _ = load_and_process_data(basin, "P", ys, ye, aggregate_time=False)
    da_et, _, _ = load_and_process_data(basin, "ET", ys, ye, aggregate_time=False)

    if da_p is None or da_et is None: return _empty_fig("Missing Data"), _empty_fig(), ""

    da_p, da_et = xr.align(da_p, da_et, join="inner")
    da_pet = da_p - da_et

    da_map = da_pet.mean(dim="time", skipna=True)
    fig_map = _create_clean_heatmap(da_map, "Mean P-ET", "RdBu", "mm")
    fig_map = add_shapefile_to_fig(fig_map, basin)

    spatial_mean = da_pet.mean(dim=["latitude", "longitude"], skipna=True)
    try:
        monthly = spatial_mean.groupby("time.month").mean(skipna=True)
        months = [pd.to_datetime(m, format="%m").strftime("%b") for m in monthly["month"].values]
        y_vals = monthly.values.flatten()
        fig_bar = px.bar(x=months, y=y_vals, title="Mean Monthly P-ET")
        fig_bar.update_traces(marker_color=THEME_COLOR)
        fig_bar.update_layout(plot_bgcolor='white', font=dict(family="Segoe UI"))
        explanation = _generate_explanation("P-ET", basin, ys, ye, y_vals, months)
    except:
        fig_bar = _empty_fig()
        explanation = ""
    return fig_map, fig_bar, dcc.Markdown(explanation, className="markdown-content")

def update_lu_map_and_coupling(basin):
    if not basin or basin == "none": return _empty_fig(), _empty_fig()

    da_lu, _, _ = load_and_process_data(basin, "LU", year_start=2020, year_end=2020)
    
    if da_lu is None: return _empty_fig("No LU Data"), _empty_fig()

    # Map
    vals = da_lu.values
    # Just return a simple heatmap for now to ensure it works
    fig_map = px.imshow(vals, origin='lower', title="Land Use")
    fig_map = add_shapefile_to_fig(fig_map, basin)

    # Bar stats
    unique, counts = np.unique(vals[np.isfinite(vals)], return_counts=True)
    total = counts.sum()
    stats = []
    for u, c in zip(unique, counts):
        name = class_info.get(int(u), {}).get("name", str(u))
        stats.append({"Class": name, "Pct": (c/total)*100})
    df_stats = pd.DataFrame(stats).sort_values("Pct", ascending=False).head(10)
    fig_bar = px.bar(df_stats, x="Pct", y="Class", orientation='h', title="Top Land Use Classes")
    fig_bar.update_traces(marker_color=THEME_COLOR)
    fig_bar.update_layout(plot_bgcolor='white', font=dict(family="Segoe UI"))

    return fig_map, fig_bar

def update_wa_module(basin, start_year, end_year):
    if not basin or basin == "none" or not start_year: return _empty_fig(), _empty_fig(), ""
    ys, ye = int(start_year), int(end_year)
    df = get_wa_data(basin, ys, ye)
    if df.empty: return _empty_fig("No Data"), _empty_fig(), "No Data"

    # Simple Sankey (Placeholder logic)
    fig_sankey = _empty_fig("Sankey Placeholder")

    # Sectoral Bar
    sector_df = df[df['CLASS'] == 'OUTFLOW'] # Simplified
    fig_bar = px.bar(sector_df, x='VARIABLE', y='VALUE', color='SUBCLASS', barmode='group')
    fig_bar.update_layout(plot_bgcolor='white', font=dict(family="Segoe UI"))

    return fig_sankey, fig_bar, html.Div("Indicators Placeholder")

def update_validation_plots(basin):
    if not basin or basin == "none": return _empty_fig(), _empty_fig()
    p_df = get_validation_data(basin, "P")
    et_df = get_validation_data(basin, "ET")

    def sc(df, t):
        if df.empty: return _empty_fig(f"No Data {t}")
        fig = px.scatter(df, x='Observed', y='Satellite', title=t)
        fig.update_traces(marker_color=THEME_COLOR)
        fig.update_layout(plot_bgcolor='white', font=dict(family="Segoe UI"))
        return fig

    return sc(p_df, "P Validation"), sc(et_df, "ET Validation")


# --- WRAPPER CALLBACKS ---

@app.callback(
    Output("basin-overview-content", "children"),
    [Input("basin-dropdown", "value"),
     Input("global-start-year-dropdown", "value"),
     Input("global-end-year-dropdown", "value")]
)
def update_basin_overview_wrapper(basin, start, end):
    return update_basin_overview(basin, start, end)

@app.callback(
    [Output("p-map-graph", "figure"), Output("p-bar-graph", "figure"), Output("p-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_p_wrapper(basin, start, end):
    return _hydro_figs(basin, start, end, "P")

@app.callback(
    [Output("et-map-graph", "figure"), Output("et-bar-graph", "figure"), Output("et-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_et_wrapper(basin, start, end):
    return _hydro_figs(basin, start, end, "ET")

@app.callback(
    [Output("p-et-map-graph", "figure"), Output("p-et-bar-graph", "figure"), Output("p-et-explanation", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_pet_wrapper(basin, start, end):
    return update_p_et_outputs(basin, start, end)

@app.callback(
    [Output("lu-map-graph", "figure"), Output("lu-bar-graph", "figure")],
    [Input("basin-dropdown", "value")]
)
def update_lu_wrapper(basin):
    # Removed year inputs as requested
    return update_lu_map_and_coupling(basin)

@app.callback(
    [Output("wa-resource-base-sankey", "figure"), Output("wa-sectoral-bar", "figure"), Output("wa-indicators-container", "children")],
    [Input("basin-dropdown", "value"), Input("global-start-year-dropdown", "value"), Input("global-end-year-dropdown", "value")]
)
def update_wa_wrapper(basin, start, end):
    return update_wa_module(basin, start, end)

@app.callback(
    [Output("val-p-scatter", "figure"), Output("val-et-scatter", "figure")],
    [Input("basin-dropdown", "value")]
)
def update_val_wrapper(basin):
    return update_validation_plots(basin)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
