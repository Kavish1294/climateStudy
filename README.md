# Climate-Change Archetypes 🌍
### Clustering cities by their *warming fingerprint*, not their temperature

An unsupervised + spatial machine-learning project on the **Berkeley Earth** surface-temperature
record. Instead of asking *"which cities are hot?"* (a question whose answer is just *latitude*),
it asks **"which cities have *changed* in the same way?"** — and then tests whether the resulting
climate-change "archetypes" form coherent regions on the map.

---

## The idea

If you cluster cities on their **average temperature**, you rediscover the equator-to-pole gradient.
That's a known result, not a finding. This project instead engineers, for every city, a compact
**fingerprint of how its climate has shifted** over the last century, clusters on *that*, and asks a
real question: **are the archetypes geographically structured, or just latitude in disguise?**

| Feature | What it captures |
|---|---|
| `warming_rate` | °C per decade trend in annual mean temperature |
| `warming_accel` | Is warming speeding up? (late-century slope − early-century slope) |
| `seasonality_amp` | Size of the summer↔winter swing |
| `seasonality_trend` | Are seasons widening or narrowing over time? |
| `volatility` | Year-to-year instability (on the deseasonalized series) |
| `volatility_trend` | Is the climate becoming more erratic? |

These describe **change**, not **level** — which is what keeps the result from collapsing back to latitude.

---

## Method

```
clean → engineer fingerprints → standardize → PCA → GMM (pick k via BIC)
      → spatial validation (η² vs latitude + Moran's I) → archetype profiles → heat map
```

1. **Feature engineering** — per-city OLS trends, seasonal amplitude, and deseasonalized volatility,
   computed only for cities with ≥ 60 years of record so the slopes are meaningful.
2. **PCA** — standardize the six features and reduce to the components covering ~90 % of variance.
3. **Clustering — Gaussian Mixture Model.** Chosen over K-Means because it fits elliptical clusters
   and gives each city a **probability** of belonging to each archetype. The number of components is
   selected by minimizing **BIC** (AIC shown alongside), and **DBSCAN** is kept as an independent
   density-based cross-check. Cities whose top probability is low are flagged as **transition cities**.
4. **Spatial validation (the core contribution).**
   - **η² ("is it just latitude?")** — variance in latitude vs. in the warming features explained by
     the cluster labels. If latitude isn't the dominant one, the archetypes carry real signal.
   - **Moran's I (from scratch)** — great-circle KNN spatial weights + a 999-run permutation test on
     `warming_rate`. Positive, significant I ⇒ warming is spatially clustered, not random.
5. **Archetype profiles** — z-scored fingerprint heatmap + highest-confidence representative cities,
   so each cluster gets a human-readable name (e.g. *"rapid accelerators"*, *"stable maritime"*).

> **Why GMM and not spatially-constrained clustering?** Methods like SKATER/max-p force clusters to be
> geographically contiguous — but then you can't use Moran's I as *independent* evidence of spatial
> structure (you'd have assumed the answer). GMM keeps the validation honest.

---

## Visualizations

- **PCA scatter** — archetypes in feature space; point size = GMM assignment confidence.
- **Interpolated heat map** — instead of one dot per city, the map paints a continuous field: at each
  point it inverse-distance-weights the GMM archetype probabilities of nearby cities and blends each
  archetype's color. Each archetype is tinted by its **average warming rate** on the `coolwarm` scale
  (blue = slower, red = faster), so the legend doubles as a warming scale. The field fades out far from
  any station, so it never invents a fingerprint over empty ocean. Renders on a real Cartopy basemap
  (coastlines + borders), with a graceful fallback to a plain lon/lat plot if Cartopy isn't installed.

---

## Repository layout

```
.
├── climate_change_archetypes.ipynb   # the project (run top-to-bottom)
└── README.md
```

## Running it

**On Kaggle (easiest).** Create a notebook, click *Add Data*, search
*"Climate Change: Earth Surface Temperature Data"* (`berkeleyearth`), and run the notebook — it
auto-detects the `/kaggle/input/...` path.

**Locally.**
```bash
pip install numpy pandas matplotlib scikit-learn cartopy   # cartopy optional but nicer
kaggle datasets download -d berkeleyearth/climate-change-earth-surface-temperature-data
unzip climate-change-earth-surface-temperature-data.zip -d data/
jupyter lab climate_change_archetypes.ipynb
```
The notebook defaults to `GlobalLandTemperaturesByMajorCity.csv` (~100 well-sampled cities). To scale to
thousands of cities, point `DATA_FILE` at `GlobalLandTemperaturesByCity.csv` — the ≥60-year filter keeps
it honest.

---

## ⚠️ Running it on the *latest* data (the Kaggle set stops in 2013)

The Kaggle dataset was packaged years ago and **ends in 2013**. Berkeley Earth still publishes monthly,
so to use current data go to **<https://berkeleyearth.org/data/>**. Two practical routes:

**Recommended — gridded NetCDF + `xarray`.** Download the land-only 1° field
`Complete_TAVG_LatLong1.nc` (~200 MB, 1753–recent). It stores monthly **anomalies** (`temperature`,
relative to 1951–1980) plus a 12-month **`climatology`**, so you reconstruct absolute monthly
temperatures and extract a series at any city's coordinates. The adapter below returns a DataFrame with
the **same columns the notebook's feature engineering already uses**, so the rest of the notebook runs
unchanged:

```python
import numpy as np, pandas as pd, xarray as xr

def _decimal_year_to_date(t):
    y = int(np.floor(t)); m = int(np.floor((t - y) * 12 + 1e-6)) + 1
    return pd.Timestamp(year=y, month=min(max(m, 1), 12), day=1)

def load_berkeley_gridded(cities, nc_path, start_year=1900, end_year=None):
    """cities: DataFrame[City, Country, lat, lon].  Returns the notebook's `df`."""
    ds = xr.open_dataset(nc_path)
    la = "latitude" if "latitude" in ds.coords else "lat"     # adjust if your file differs
    lo = "longitude" if "longitude" in ds.coords else "lon"   # (check with `ds` or `ncdump -h`)
    times = pd.DatetimeIndex([_decimal_year_to_date(float(t)) for t in ds["time"].values])
    out = []
    for _, c in cities.iterrows():
        pt   = ds.sel({la: c.lat, lo: c.lon}, method="nearest")
        anom = np.asarray(pt["temperature"].values, float)    # monthly anomaly
        clim = np.asarray(pt["climatology"].values, float)    # 12-month seasonal normal
        out.append(pd.DataFrame({
            "dt": times, "AverageTemperature": anom + clim[times.month.values - 1],
            "City": c.City, "Country": c.Country, "lat": float(c.lat), "lon": float(c.lon)}))
    df = pd.concat(out, ignore_index=True).dropna(subset=["AverageTemperature"])
    df["year"], df["month"] = df.dt.dt.year, df.dt.dt.month
    df["city_id"] = df.City + ", " + df.Country
    end_year = end_year or int(df.year.max())
    return df[(df.year >= start_year) & (df.year <= end_year)].copy()

# cities = pd.DataFrame({"City":[...], "Country":[...], "lat":[...], "lon":[...]})
# df = load_berkeley_gridded(cities, "Complete_TAVG_LatLong1.nc")
# --> then SKIP the notebook's coordinate-parsing step and go straight to feature engineering.
```
Because Berkeley Earth reports anomalies, the trend/volatility features work directly; the
`climatology` term is what lets the **seasonality** features keep working. For a fast first pass, grab a
small regional file (e.g. `Europe_TAVG_Gridded_1.nc`, ~15 MB) instead of the global one.

**Alternative — per-city text files.** Berkeley Earth also offers ~110 km city estimates
(<https://berkeleyearth.org/temperature-city-list/>) and an ML-augmented **~25 km, ~8,000-city** set
*by request*. These are fixed-width `.txt` files with a `%`-comment header — closer to the original
Kaggle packaging, but you'll write a small parser and the seasonal cycle lives in the header.

---

## Caveats

- **Kaggle data ends 2013.** Treat absolute warming rates as illustrative; re-run on current Berkeley
  Earth data (above) for up-to-date figures.
- **The heat map is interpolated** from sparse, unevenly spaced stations — read it as *"where each
  archetype dominates,"* not a physical reanalysis.
- **Heat-map colors are relative** (min–max across archetypes), which is why each legend entry also
  prints the actual °C/decade. For an absolute scale (white pinned at 0), swap `plt.Normalize` for
  `matplotlib.colors.TwoSlopeNorm(vcenter=0, ...)`.
- **GMM covariance:** `covariance_type="full"` is fine at this scale; on very small subsets switch to
  `"diag"` if a component's covariance becomes unstable.

## Possible extensions

1. Scale to the full by-city file (thousands of stations).
2. Swap PCA → UMAP for the 2-D view.
3. Shade the map by GMM `confidence` to reveal climate transition zones.
4. Uncertainty-weighted trend fits using Berkeley Earth's reported uncertainties.
5. Train a classifier to predict a city's archetype from geography — how much is geographically determined?

## Data & license

Data © **Berkeley Earth**, released under **CC BY-NC 4.0** (non-commercial; attribute Berkeley Earth and
<https://www.berkeleyearth.org>). Source: <https://berkeleyearth.org/data/> and the Kaggle mirror
*Climate Change: Earth Surface Temperature Data*.
