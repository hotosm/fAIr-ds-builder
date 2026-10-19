# Load a dataset

Each dataset is a Hugging Face dataset with `train`, `validation` and `test` splits. A row is
one 256x256 image tile with the labels that fall on it.

## Load the rows

```python
from datasets import load_dataset

dataset = load_dataset("hotosm/fAIr-guru-td-buildings")
print(dataset)
```

The main columns:

| Column | Content |
|--------|---------|
| `image` | The tile as served by OpenAerialMap; its bounds are in the `bbox_*` columns |
| `mask` | Binary mask, 0 background and 255 feature, in the buildings and highways datasets. Empty in the others |
| `label_geojson` | GeoJSON FeatureCollection of the labels on the tile, in EPSG:4326, with their OSM tags |
| `class_counts` | JSON object counting each class on the tile |
| `num_features` | Number of labels on the tile |
| `feature_type` | The OSM tag key whose value is the class, such as `building` or `roof:material` |
| `tile_x`, `tile_y`, `tile_z` | XYZ tile of the row |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | Tile bounds in EPSG:4326 |
| `project_id`, `project_url`, `country` | The Tasking Manager project the tile comes from |

## Read labels and classes

`label_geojson` and `class_counts` are JSON strings.

```python
import json

row = dataset["train"][0]
labels = json.loads(row["label_geojson"])
for feature in labels["features"]:
    print(feature["properties"]["tags"][row["feature_type"]], feature["geometry"]["type"])
```

The class of a label is the value of its `feature_type` tag: `yes` or `house` for
buildings, `metal` or `concrete` for roof material, `asphalt` or `dirt` for road surface.
Each card states how many tiles with `num_features` 0 it holds.

## Build masks from labels

The roof type, road surface, trees and solar datasets ship labels only, with an empty `mask` column. The
`bbox_*` columns give each tile's bounds, which place its labels on the image pixels, so the
labels can be burned into a mask with rasterio. Fix the class list once, from the class table
on the dataset card, so a class keeps the same mask value on every tile:

```python
import json

from datasets import load_dataset
from rasterio.features import rasterize
from rasterio.transform import from_bounds

CLASSES = ["metal", "metal_sheet", "concrete", "roof_tiles"]

dataset = load_dataset("hotosm/fAIr-guru-td-roof-type", split="train")
labelled = dataset.filter(lambda count: count > 0, input_columns="num_features")
row = labelled[0]
image = row["image"]
labels = json.loads(row["label_geojson"])

bounds = (row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"])
transform = from_bounds(*bounds, image.width, image.height)
shapes = [
    (feature["geometry"], CLASSES.index(feature["properties"]["tags"]["roof:material"]) + 1)
    for feature in labels["features"]
    if feature["properties"]["tags"]["roof:material"] in CLASSES
]
mask = rasterize(shapes, out_shape=(image.height, image.width), transform=transform)
```

Points burn as single pixels and lines as one-pixel-wide lines. Buffer them first, for
example road centrelines by half a road width or tree points by their `diameter_crown`, to
get areas.

## Splits

The `train`, `validation` and `test` splits together hold every tile. They are spatial:
each project is divided into contiguous regions, and each region goes whole to one split.
Tiles at a region edge border tiles of another split, so scores on the published held-out
splits are optimistic.
[Make your own split](own-split.md) explains how to build one for model evaluation.
