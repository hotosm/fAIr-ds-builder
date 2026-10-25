---
language:
  - en
license:
  - cc-by-4.0
  - odbl
task_categories:
  - image-segmentation
task_ids:
  - semantic-segmentation
tags:
  - remote-sensing
  - aerial-imagery
  - openstreetmap
  - openaerialmap
  - humanitarian
  - geospatial
pretty_name: HOT OAM Dataset
dataset_info:
  features:
    - name: image
      dtype: image
    - name: mask
      dtype: image
    - name: tile_id
      dtype: string
    - name: tile_x
      dtype: int32
    - name: tile_y
      dtype: int32
    - name: tile_z
      dtype: int32
    - name: project_id
      dtype: int32
    - name: project_name
      dtype: string
    - name: country
      dtype: string
    - name: organisation
      dtype: string
    - name: imagery_url
      dtype: string
    - name: feature_type
      dtype: string
    - name: num_features
      dtype: int32
    - name: label_geojson
      dtype: large_string
    - name: bbox_west
      dtype: float64
    - name: bbox_south
      dtype: float64
    - name: bbox_east
      dtype: float64
    - name: bbox_north
      dtype: float64
  splits:
    - name: train
      num_examples: 0
---

<!-- DESCRIPTION_START -->
# HOT OAM Dataset

Training data for segmentation and object detection. Aerial imagery tiles from
OpenAerialMap are paired with OpenStreetMap labels, drawn from HOT Tasking Manager projects.

<!-- AUTO_STATS_START -->
<!-- AUTO_STATS_END -->
<!-- DESCRIPTION_END -->

## How it was labelled

The labels are [OpenStreetMap](https://www.openstreetmap.org/) features mapped in
[HOT Tasking Manager](https://tasks.hotosm.org/) projects
([wiki](https://wiki.openstreetmap.org/wiki/Tasking_Manager)). Tasking Manager divides a
project area into small tasks, so many volunteers can map it together following the project
instructions. A mapper traces the features in a task, and a validator then checks the
task and marks it validated. Coordinated mapping of this kind falls under the OpenStreetMap
Foundation's [Organised Editing Guidelines](https://osmfoundation.org/wiki/Organised_Editing_Guidelines).

The imagery comes from [OpenAerialMap](https://openaerialmap.org/about/)
([wiki](https://wiki.openstreetmap.org/wiki/OpenAerialMap)), where people and organisations
share openly licensed satellite, aerial and drone imagery through the Open Imagery Network.
Each project sets this OpenAerialMap imagery for its mappers, and the tiles come from it.

<selection>

## Splits

The splits are spatial. Each project's tiles are grouped into contiguous regions, and each
region goes whole to train, validation or test, as listed under Split method above. For model
evaluation we recommend making your own split, as described in
[Make your own split](https://hotosm.github.io/fAIr-ds-builder/guides/own-split/).

![Train, validation and test tiles per project](https://huggingface.co/datasets/<repo>/resolve/v<version>/stac/split_map.png)

## Load

```python
import json

from datasets import load_dataset
from rasterio.features import rasterize
from rasterio.transform import from_bounds

dataset = load_dataset("<repo>", split="train")
labelled = dataset.filter(lambda count: count > 0, input_columns="num_features")
row = labelled[0]
image = row["image"]
labels = json.loads(row["label_geojson"])

bounds = (row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"])
transform = from_bounds(*bounds, image.width, image.height)
shapes = [feature["geometry"] for feature in labels["features"]]
mask = rasterize(shapes, out_shape=(image.height, image.width), transform=transform)
```

`image` is the 256x256 tile as served by OpenAerialMap. The `bbox_*` columns hold its bounds,
which place `label_geojson` on its pixels.

| Column | Description |
|--------|-------------|
| `image` | RGB tile |
| `mask` | Binary raster of the labels, 0 background and 255 feature, in datasets built with masks |
| `label_geojson` | Labels in the tile with their OpenStreetMap tags, EPSG:4326 |
| `num_features`, `class_counts` | Label count, and the count per class as JSON |
| `tile_id`, `tile_x`, `tile_y`, `tile_z` | XYZ tile identity |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | Tile bounds, EPSG:4326 |
| `project_id`, `project_name`, `project_url`, `country`, `organisation` | Source Tasking Manager project |
| `imagery_url`, `feature_type` | OpenAerialMap tile URL, and the OpenStreetMap key of the classes |

## Browse the STAC catalogue

Each project is one item holding a Cloud Optimized GeoTIFF of its tiles, the labels and the
tile footprints with their split.
[Open in STAC Browser](https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets/<repo>/raw/v<version>/stac/collection.json)
to see the labels over the imagery, or follow the
[QGIS guide](https://hotosm.github.io/fAIr-ds-builder/guides/qgis/) to open them in QGIS.

## Known limitations

<limitations>

## Sources and licensing

| Component | Source | License |
|-----------|--------|---------|
| Imagery | [OpenAerialMap](https://openaerialmap.org/) | CC-BY 4.0, attribute contributors of Open Imagery Network |
| Labels | [OpenStreetMap](https://www.openstreetmap.org/) | ODbL 1.0, attribute OpenStreetMap contributors |

Built with [fAIr-ds-builder](https://github.com/hotosm/fAIr-ds-builder).
