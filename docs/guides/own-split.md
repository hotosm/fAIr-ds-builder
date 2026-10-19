# Make your own split

Each dataset ships a spatial split: every Tasking Manager project is divided into contiguous
regions, and one region goes to validation, one to test and the rest to train. Every tile is
kept, so the train, validation and test splits
on Hugging Face hold all the data. Tiles along a region edge sit next to tiles of another
split, so scores measured on the published validation and test splits are optimistic. For
model evaluation, make a split that matches where the model will be used.

## Why the split matters

Neighbouring tiles look alike and carry similar labels. When validation tiles sit next to
training tiles, a model is scored on ground it has nearly seen, and the score overstates
how it performs on a new area.

- Roberts et al. (2017) found that ignoring spatial structure in cross-validation seriously
  underestimates prediction error, and recommend blocks substantially larger than the range
  of spatial autocorrelation ([doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)).
- Kattenborn et al. (2022) found that random hold-outs overestimated CNN performance on
  drone tree segmentation by up to 28%, and that autocorrelation is significantly higher
  within one image acquisition than between acquisitions
  ([doi:10.1016/j.ophoto.2022.100018](https://doi.org/10.1016/j.ophoto.2022.100018)).
- Abriha et al. (2023) found random splits inflated roof material accuracy by up to 17%
  ([Heliyon](https://pmc.ncbi.nlm.nih.gov/articles/PMC10006495/)).
- Linnenbrink et al. (2024) match the distance between training and test tiles to the
  distance between training data and the area the model will be applied to
  ([doi:10.5194/gmd-17-5897-2024](https://doi.org/10.5194/gmd-17-5897-2024)).
- Wadoux et al. (2021) argue that for the accuracy of a map inside the sampled area,
  probability sampling is the right tool and spatial cross-validation is not
  ([doi:10.1016/j.ecolmodel.2021.109692](https://doi.org/10.1016/j.ecolmodel.2021.109692)).

## Choose a separation distance

Neighbouring tiles stay alike up to some distance, the range of spatial autocorrelation. The
range of a variogram fitted to the tiles' label counts or image values estimates it, for
example with [scikit-gstat](https://scikit-gstat.readthedocs.io/). Separate held-out areas
from training areas by at least that range.
When a project is too small for a buffer of that size, hold out whole projects, or whole
image acquisitions, where the dataset has several.

## Build a split with a buffer

The tile bounds in the `bbox_*` columns are enough to cluster tiles into regions and leave a
buffer between them:

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

The distance is measured between tile footprints, so a buffer of 900 m leaves at least 900 m
of ground between a training tile and a held-out tile. Run it per project when a dataset
spans several, and choose `buffer_meters` from the ranges on the card.

## Build a split with this tool

The `split` section of a dataset config sets the same rules when a dataset is built:

```yaml
split:
  regions_per_project: 10
  validation_regions: 1
  test_regions: 1
  buffer_meters: 900
  exclude_shared_features: true
```

`buffer_meters` leaves out validation and train tiles within that distance of a test tile,
and train tiles within that distance of a validation tile. `exclude_shared_features` also
leaves out tiles holding part of a feature, such as a long road, that lies in a higher split.
Tiles left out this way stay in the raw files and are counted on the dataset card.
