# Inspect in QGIS

QGIS reads the STAC catalogue of a dataset from its Browser panel, so the imagery, labels and
splits of one Tasking Manager project can be loaded together and inspected feature by
feature.

## Add the connection

1. In the **Browser** panel, right-click **STAC** and choose **New STAC Connection**.
2. Give it a name and enter the collection URL of the dataset:

    ```text
    https://huggingface.co/datasets/hotosm/fAIr-guru-td-buildings/resolve/main/stac/collection.json
    ```

3. Expand the connection. It lists two items per Tasking Manager project: the labels item,
   with its `labels` (GeoJSON), `labels_parquet` and `tiles` assets, and the imagery item,
   with its `imagery` asset.

![Browser panel with the STAC connection and one project expanded](../assets/qgis/browser.png)

## Load a project

Drag the project's imagery item onto the map to add the imagery. QGIS adds STAC assets in the
COG, COPC, EPT, Zarr and Parquet formats from version 4.0, so there dragging the labels item
adds the GeoParquet labels and tiles
([source](https://github.com/qgis/QGIS/blob/release-4_0/src/core/stac/qgsstacasset.cpp)).
QGIS 3.44 adds only COG, COPC and EPT assets
([source](https://github.com/qgis/QGIS/blob/release-3_44/src/core/stac/qgsstacasset.cpp));
add the labels and tiles with **Layer > Add Layer > Add Vector Layer**, choose **Protocol:
HTTP(S), cloud, etc.**, and enter the asset URL, for example:

```text
https://huggingface.co/datasets/hotosm/fAIr-guru-td-buildings/resolve/main/stac/project-40695/labels.parquet
```

## Show the splits

Open the properties of the tiles layer, choose **Symbology**, then **Categorized** with
**Value** `split`, and **Classify**. Each tile footprint is coloured by its split, and the
contiguous held-out regions stand out from the training area.

![Imagery, labels and tiles coloured by split for project 40695](../assets/qgis/overview.png)

Zoom in to compare the labels with the imagery.

![Labels over the imagery at tile scale](../assets/qgis/closeup.png)

## Read the attributes

Use **Identify Features** on a label or a tile, or open the layer's attribute table.

A label holds its tile, split, project, class, OSM id and type, the time of its last OSM
edit, and its OSM tags:

![Attributes of one label](../assets/qgis/label-attributes.png)

A tile footprint holds its split, its label count and its classes:

![Attributes of one tile footprint](../assets/qgis/tile-attributes.png)

The **Details** action on a labels item or on the connection shows the STAC metadata: the
tile counts per split, the label type and description, and the Tasking Manager project.
QGIS 3.44 leaves fields holding lists blank there, such as `label:classes`, `label:tasks` and
`tm:mapping_types`; STAC Browser shows them in full.

![STAC item details of project 40695](../assets/qgis/item-details.png)
