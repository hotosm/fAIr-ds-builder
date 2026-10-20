# tm-oam-ds-builder

Training datasets from [HOT Tasking Manager](https://tasks.hotosm.org/) projects: 256x256
[OpenAerialMap](https://openaerialmap.org/) image tiles with the
[OpenStreetMap](https://www.openstreetmap.org/) features mapped on them, published on Hugging
Face with a [STAC](https://stacspec.org/) catalogue.

To use a dataset, see [Load a dataset](guides/load-a-dataset.md), or view it on a map with
[STAC Browser](guides/stac-browser.md) or [QGIS](guides/qgis.md). To make one from your own
Tasking Manager projects, see [Build a dataset](guides/build-a-dataset.md).

## Datasets

From the [fAIr Open Mapping Gurus](https://hotosm.github.io/fAIr/datasets/#open-mapping-gurus)
programme, where community mappers create training data for fAIr, all in one [STAC catalogue](https://huggingface.co/datasets/hotosm/fAIr-guru-stac)
([open in STAC Browser](https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json)).

| Dataset | Labels | Tasks | Zoom | Projects |
|---------|--------|-------|------|----------|
| [fAIr-guru-td-buildings](https://huggingface.co/datasets/hotosm/fAIr-guru-td-buildings) | Building footprints, with masks | Segmentation | 19 | 40695, 50793, 60363 |
| [fAIr-guru-td-highways](https://huggingface.co/datasets/hotosm/fAIr-guru-td-highways) | Road centrelines, with 6 m masks | Segmentation | 19 | 40695, 50793, 60363 |
| [fAIr-guru-td-roof-type](https://huggingface.co/datasets/hotosm/fAIr-guru-td-roof-type) | Buildings by `roof:material` | Segmentation, classification | 19 | 40695, 50793 |
| [fAIr-guru-td-road-surface](https://huggingface.co/datasets/hotosm/fAIr-guru-td-road-surface) | Roads by `surface` | Segmentation, classification | 19 | 40695, 50793, 60363 |
| [fAIr-guru-td-trees](https://huggingface.co/datasets/hotosm/fAIr-guru-td-trees) | Tree points and tree groups | Detection | 20 | 50661 |
| [fAIr-guru-td-solar](https://huggingface.co/datasets/hotosm/fAIr-guru-td-solar) | Rooftop solar panels | Segmentation | 21 | 50661 |

## Licensing

Imagery CC-BY 4.0, "contributors of Open Imagery Network" ([source](https://openaerialmap.org/about/)).
Labels ODbL 1.0, "OpenStreetMap contributors" ([source](https://www.openstreetmap.org/copyright)).
Tooling GPL-3.0-or-later.
