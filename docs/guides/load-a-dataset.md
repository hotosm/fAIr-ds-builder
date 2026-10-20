# Load a dataset

Each dataset has `train`, `validation` and `test` splits. A row is one 256x256 tile with its
labels.

Install the Hugging Face `datasets` library with `pip install datasets`. The mask example below
also needs `rasterio`.

```python
from datasets import load_dataset

dataset = load_dataset("hotosm/fAIr-guru-td-buildings")
```

| Column | Content |
|--------|---------|
| `image` | The tile as served by OpenAerialMap |
| `mask` | Binary mask (0 background, 255 feature) in the buildings and highways datasets |
| `label_geojson` | GeoJSON of the labels on the tile, in longitude and latitude (EPSG:4326), with their OSM tags |
| `class_counts` | JSON count of each class on the tile |
| `num_features` | Number of labels on the tile |
| `feature_type` | OSM tag key whose value is the class |
| `tile_id`, `tile_x`, `tile_y`, `tile_z` | [XYZ tile](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | Tile bounds in EPSG:4326 |
| `project_id`, `project_name`, `project_url`, `country`, `organisation` | Source Tasking Manager project |
| `imagery_url` | OpenAerialMap image the tile comes from |

## Read the labels

```python
import json

row = dataset["train"][0]
for feature in json.loads(row["label_geojson"])["features"]:
    print(feature["properties"]["tags"][row["feature_type"]], feature["geometry"]["type"])
```

## Make masks from labels

The roof type, road surface, trees and solar datasets carry labels only. Burn a mask from them
with a fixed class list from the card:

```python
import json

from datasets import load_dataset
from rasterio.features import rasterize
from rasterio.transform import from_bounds

CLASSES = ["metal", "metal_sheet", "concrete", "roof_tiles"]

dataset = load_dataset("hotosm/fAIr-guru-td-roof-type", split="train")
row = dataset.filter(lambda count: count > 0, input_columns="num_features")[0]
image = row["image"]
bounds = (row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"])
transform = from_bounds(*bounds, image.width, image.height)
shapes = [
    (feature["geometry"], CLASSES.index(feature["properties"]["tags"]["roof:material"]) + 1)
    for feature in json.loads(row["label_geojson"])["features"]
    if feature["properties"]["tags"]["roof:material"] in CLASSES
]
mask = rasterize(shapes, out_shape=(image.height, image.width), transform=transform)
```

Buffer points and lines first to get areas, for example roads by half their width.

## Splits

Held-out tiles at a region edge sit next to training tiles, so scores are optimistic. See
[Make your own split](own-split.md).
