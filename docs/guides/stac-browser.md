# Browse with STAC Browser

[Open the root catalogue in STAC Browser](https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json).

It lists one release of each dataset. Each dataset is a STAC Collection with one item per
Tasking Manager project, holding its imagery, labels and tiles. A single release opens the same
way, for example [fAIr-guru-td-buildings
v0.4.1](https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets/hotosm/fAIr-guru-td-buildings/raw/v0.4.1/stac/collection.json).

![Root catalogue with the six datasets](../assets/stac-browser/start.png)

![Collection page of fAIr-guru-td-buildings](../assets/stac-browser/collection.png)

## A project

An item shows its labels on the map, class counts, tiles per split and the Tasking Manager
project. **Show on map** on the imagery asset draws the imagery, with its capture date and
pixel size (`raster:spatial_resolution`). [QGIS](qgis.md) shows imagery and labels together.

![Item of project 40695 with its labels on the map](../assets/stac-browser/item.png)

Large label files, such as the 120,000 trees of project 50661, take a while to draw.

## Assets

The item holds the imagery (Cloud Optimized GeoTIFF), the labels
([GeoJSON](https://geojson.org/) and [GeoParquet](https://geoparquet.org/)) and the tile
footprints with their split (GeoParquet). **Open in geoparquet.info** shows a GeoParquet file.
To colour tiles by split, use [QGIS](qgis.md).

![Labels assets with their download and geoparquet.info links](../assets/stac-browser/labels-asset.png)
