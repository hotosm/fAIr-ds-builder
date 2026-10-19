# tm-oam-ds-builder

tm-oam-ds-builder builds machine learning training datasets from
[HOT Tasking Manager](https://tasks.hotosm.org/) projects. Each dataset pairs 256x256 image
tiles from [OpenAerialMap](https://openaerialmap.org/) with the
[OpenStreetMap](https://www.openstreetmap.org/) features mapped and validated in those
projects, and is published on Hugging Face with a STAC catalogue for GIS tools.

## Datasets

The datasets of the [fAIr Open Mapping Gurus](https://hotosm.github.io/fAIr/datasets/#open-mapping-gurus)
programme are built from the configs in
[`configs/`](https://github.com/hotosm/tm-oam-ds-builder/tree/master/configs).

| Dataset | Labels | Tasks | Zoom | Tasking Manager projects |
|---------|--------|------|------|--------------------------|
| [fAIr-guru-td-buildings](https://huggingface.co/datasets/hotosm/fAIr-guru-td-buildings) | Building footprints, with masks | Segmentation | 19 | 40695, 50793 |
| [fAIr-guru-td-highways](https://huggingface.co/datasets/hotosm/fAIr-guru-td-highways) | Road centrelines, with 6 m masks | Segmentation | 19 | 40695, 50793 |
| [fAIr-guru-td-roof-type](https://huggingface.co/datasets/hotosm/fAIr-guru-td-roof-type) | Building footprints classed by `roof:material` | Segmentation, classification | 19 | 40695, 50793 |
| [fAIr-guru-td-road-surface](https://huggingface.co/datasets/hotosm/fAIr-guru-td-road-surface) | Road centrelines classed by `surface` | Segmentation, classification | 19 | 40695, 50793 |
| [fAIr-guru-td-trees](https://huggingface.co/datasets/hotosm/fAIr-guru-td-trees) | Tree points and tree group polygons | Detection | 20 | 50661 |
| [fAIr-guru-td-solar](https://huggingface.co/datasets/hotosm/fAIr-guru-td-solar) | Rooftop solar panel outlines | Segmentation | 21 | 50661 |

Each dataset card on Hugging Face lists its tile, label and class counts, its splits, and its
known limitations.

## Guides

- [Load a dataset](guides/load-a-dataset.md) in Python and turn its labels into masks.
- [Make your own split](guides/own-split.md) for model evaluation, with a buffer between training and held-out areas.
- [Browse with STAC Browser](guides/stac-browser.md) to see the tiles, splits and labels on a map in the web browser.
- [Inspect in QGIS](guides/qgis.md) to load the imagery, labels and splits of a project and read their attributes.
- [Build a dataset](guides/build-a-dataset.md) from Tasking Manager projects with a config file.

## Licensing

Imagery is CC-BY 4.0, with attribution to contributors of Open Imagery Network
([source](https://openaerialmap.org/about/)). Labels are ODbL 1.0, with attribution to
OpenStreetMap contributors ([source](https://www.openstreetmap.org/copyright)). The tooling
is GPL-3.0-or-later.
