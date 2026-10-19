# Browse with STAC Browser

Each dataset ships a static [STAC](https://stacspec.org/) Collection in its `stac/` folder.
Each Tasking Manager project has a labels item and an imagery item, laid out as the
[STAC label extension](https://github.com/stac-extensions/label) describes.
[STAC Browser](https://github.com/radiantearth/stac-browser) shows them on a map in the web
browser, with nothing to install.

## Open the collection

1. Open [radiantearth.github.io/stac-browser](https://radiantearth.github.io/stac-browser/).
2. Paste the collection URL of a dataset into the box at the top and choose **Load**:

    ```text
    https://huggingface.co/datasets/hotosm/fAIr-guru-td-buildings/resolve/main/stac/collection.json
    ```

    Replace `fAIr-guru-td-buildings` with the dataset you want.

![STAC Browser start page with the URL box](../assets/stac-browser/start.png)

The collection page lists its description, license, extent and one item per project.

![Collection page of fAIr-guru-td-buildings](../assets/stac-browser/collection.png)

## Open a project

Choose a project in the item list; the imagery items are listed next to them with
`imagery` in their names. A project's labels item page shows:

- the labels drawn over the imagery of its imagery item, which STAC Browser loads through the
  item's `source` link
- the labels section of the label extension: type, classes, tasks, method and the count per
  class
- the dataset properties: zoom, tiles per split and label count
- the Tasking Manager project: id, organisation, mapping types and country, with a link to
  the project page

The imagery item behind the `source` link shows the imagery's capture date and the ground pixel
size in metres at the project's centre latitude (`raster:spatial_resolution`). The collection page links the OpenStreetMap wiki page of each
tag the labels were selected on.

![Labels item of project 40695 with its labels over the imagery](../assets/stac-browser/item.png)

The labels GeoJSON of a large project, such as the 120,000 trees of project 50661, takes a
while to download before it is drawn.

## Labels and tiles

The labels item has three assets: the labels as GeoJSON, the same labels as GeoParquet, and
the tile footprints with their split as GeoParquet. STAC Browser lists each with a download
link, and the GeoParquet ones with **Open in geoparquet.info**, which opens the file in a web
viewer. To style the tiles by split over the imagery, open the items in QGIS as described in
[Inspect in QGIS](qgis.md).

![Labels assets with their download and geoparquet.info links](../assets/stac-browser/labels-asset.png)
