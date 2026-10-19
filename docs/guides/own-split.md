# Make your own split

The published split keeps every tile, so held-out tiles at a region edge sit next to similar
training tiles and scores are optimistic. For evaluation, leave a buffer between them.

- Roberts et al. (2017): ignoring spatial structure underestimates prediction error; use blocks larger than the autocorrelation range ([doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)).
- Kattenborn et al. (2022): random hold-outs overestimated CNN tree segmentation by up to 28% ([doi:10.1016/j.ophoto.2022.100018](https://doi.org/10.1016/j.ophoto.2022.100018)).
- Abriha et al. (2023): random splits inflated roof material accuracy by up to 17% ([Heliyon](https://pmc.ncbi.nlm.nih.gov/articles/PMC10006495/)).

Set the buffer to the autocorrelation range, for example from a variogram of label counts with
[scikit-gstat](https://scikit-gstat.readthedocs.io/). For projects smaller than that, hold out
whole projects.

## In Python

```python
import geopandas as gpd
import numpy as np
from datasets import concatenate_datasets, load_dataset
from shapely.geometry import box
from sklearn.cluster import KMeans

parts = load_dataset("hotosm/fAIr-guru-td-trees")
columns = ["tile_id", "project_id", "bbox_west", "bbox_south", "bbox_east", "bbox_north"]
frame = concatenate_datasets(list(parts.values())).select_columns(columns).to_pandas()

footprints = gpd.GeoSeries(
    [box(r.bbox_west, r.bbox_south, r.bbox_east, r.bbox_north) for r in frame.itertuples()],
    crs=4326,
)
footprints = footprints.to_crs(footprints.estimate_utm_crs())
centres = np.column_stack([footprints.centroid.x, footprints.centroid.y])
frame["region"] = KMeans(10, n_init=10, random_state=0).fit_predict(centres)
frame["split"] = frame.region.map({0: "test", 1: "validation"}).fillna("train")

buffer_meters = 900
near_test = footprints.distance(footprints[frame.split == "test"].union_all()) < buffer_meters
near_validation = (
    footprints.distance(footprints[frame.split == "validation"].union_all()) < buffer_meters
)
keep = (
    (frame.split == "test")
    | ((frame.split == "validation") & ~near_test)
    | ((frame.split == "train") & ~near_test & ~near_validation)
)
frame = frame[keep.to_numpy()]
```

Run it per project when a dataset spans several.

## With this tool

Set the buffer in the config's `split` section, then run `--generate-splits`,
`--generate-parquet` and `--generate-stac`:

```yaml
split:
  regions_per_project: 10
  validation_regions: 1
  test_regions: 1
  buffer_meters: 900
  exclude_shared_features: true
```
