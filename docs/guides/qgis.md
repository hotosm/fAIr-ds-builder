# Inspect in QGIS

## Connect

1. In the **Browser** panel, right-click **STAC** and choose **New STAC Connection**.
2. Enter the root catalogue URL:

    ```text
    https://huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json
    ```

3. Expand a dataset. Each Tasking Manager project is one item.

![Browser panel with the STAC connection and one project expanded](../assets/qgis/browser.png)

## Load a project

Drag the project's item onto the map. From QGIS 4.0 this adds the imagery, and the GeoParquet
labels and tiles
([source](https://github.com/qgis/QGIS/blob/release-4_0/src/core/stac/qgsstacasset.cpp)). In
QGIS 3.44 it adds the imagery
([source](https://github.com/qgis/QGIS/blob/release-3_44/src/core/stac/qgsstacdataitems.cpp)); add
the labels and tiles with **Layer > Add Layer > Add Vector Layer**, **Protocol: HTTP(S)**, and
the asset URL.

## Show the splits

In the tiles layer's **Symbology**, choose **Categorized** by `split` and **Classify**.

![Imagery, labels and tiles coloured by split for project 40695](../assets/qgis/overview.png)

![Labels over the imagery at tile scale](../assets/qgis/closeup.png)

## Read the attributes

**Identify Features** on a label shows its tile, split, project, class, OSM id and tags:

![Attributes of one label](../assets/qgis/label-attributes.png)

A tile shows its split, label count and classes:

![Attributes of one tile footprint](../assets/qgis/tile-attributes.png)

**Details** on an item shows its STAC metadata. QGIS 3.44 and 4.2 leave list fields such as
`label:classes` blank there.

![STAC item details of project 40695](../assets/qgis/item-details.png)
