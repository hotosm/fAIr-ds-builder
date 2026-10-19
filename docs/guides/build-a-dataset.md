# Build a dataset

A dataset is described by one YAML config, like those in
[`configs/`](https://github.com/hotosm/tm-oam-ds-builder/tree/master/configs), and built with
the `hot-oam-dataset` command.

## Install

```bash
uv tool install git+https://github.com/hotosm/tm-oam-ds-builder
```

Private Tasking Manager projects need a session token in the `TM_TOKEN` environment variable.

## Write the config

Keys are the long command line options with underscores. `split` sets the train,
validation and test split, `card` holds the dataset card text written by hand, and `fair`
names who registers the dataset in fAIr:

```yaml
output_dir: fAIr-guru-td-trees
project_ids: 50661
feature_type: natural
tag_values: [tree, tree_group]
geometry_types: [point, polygon]
zoom: 20
burn_binary_masks: false
dataset_version: 0.1.0
split:
  regions_per_project: 10
  validation_regions: 1
  test_regions: 1
  buffer_meters: 0
  exclude_shared_features: false
card:
  repo_id: hotosm/fAIr-guru-td-trees
  title: "fAIr Open Mapping Gurus: trees"
  label_noun: trees
  tasks: [detection]
  summary: >-
    Tree training data from a Freetown, Sierra Leone project.
  selection: >-
    Labels are every OpenStreetMap feature with natural=tree or natural=tree_group.
  limitations:
    - diameter_crown is a mapper estimate.
fair:
  user_id: hot
  provider: hot
```

| Key | Effect |
|-----|--------|
| `output_dir` | Folder the dataset is built in |
| `dataset_version` | Version written to the card, the Parquet and the Hub tag |
| `project_ids` | Tasking Manager projects to build. Without it, validated OpenAerialMap projects are discovered |
| `feature_type` | OSM tag key to label. Its value is each label's class |
| `tag_values` | Keep only these values of the tag |
| `require_tags` | Keep only features that also carry these keys, such as `highway` for road surfaces |
| `geometry_types` | `point`, `line` or `polygon` features to request |
| `tm_mapping_type` | Tasking Manager mapping type the projects are checked against |
| `zoom` | Tile zoom level |
| `burn_binary_masks` | Write a binary mask per tile |
| `label_buffer_meters` | Widen points and lines before masks are burned |
| `negative_ratio` | Keep at most this many tiles without labels per labelled tile, per project |
| `split` | Contiguous regions per project held out for validation and test, an optional buffer and shared-feature rule. See [Make your own split](own-split.md) |
| `card` | Title, summary, selection, label noun, tasks and limitations of the dataset card. `tasks` lists `segmentation`, `detection` or `classification` and also sets `label:tasks` in the STAC items |
| `fair` | `user_id` and `provider` written on the fAIr dataset item |

## Build

```bash
hot-oam-dataset --config fair-guru-trees.yaml --all
```

This downloads the tiles and labels, assigns the splits, writes the Parquet and dataset card,
and writes the STAC catalogue. The card layout ships with the package, so the config is the
only file the build needs. The card's stats and source project table are computed from
the rows written to the Parquet.

## Publish

```bash
hot-oam-dataset --config fair-guru-trees.yaml --push-to-hub hotosm/fAIr-guru-td-trees
```

The push uploads the Parquet with the images embedded, the card as the repo README, and the
STAC catalogue, and tags the revision `v<dataset_version>`. The STAC links on the Hub point at
that tag. The push stops when the card was written for a different dataset version, and when
that version was already pushed: raise `dataset_version` in the config for every new push.

## Register in fAIr

```bash
uv tool install "hot-oam-dataset[fair] @ git+https://github.com/hotosm/tm-oam-ds-builder"
export FAIR_STAC_API_KEY=<key>
hot-oam-dataset --config fair-guru-trees.yaml --publish-stac https://stac.ai.hotosm.org/stac
```

This writes one item to the `datasets` collection of fAIr's STAC API, built with fAIr's own
[fair-py-ops](https://github.com/hotosm/fAIr-models). Run it after the push: it needs the
Hub repo's tag for the dataset version, because the item's `chips` and `labels` assets
point at that tag's Parquet (`hf://datasets/<repo>@v<version>/data`).
