# fAIr-ds-builder

Builds machine learning training datasets from [HOT Tasking Manager](https://tasks.hotosm.org/)
projects. Each dataset pairs 256x256 [OpenAerialMap](https://openaerialmap.org/) image tiles
with the [OpenStreetMap](https://www.openstreetmap.org/) features mapped on them, and is
published on Hugging Face with a STAC catalogue.

Guides: [hotosm.github.io/fAIr-ds-builder](https://hotosm.github.io/fAIr-ds-builder/). The
repository is `fAIr-ds-builder`, and the command it installs is `hot-oam-dataset`.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/hotosm/fAIr-ds-builder
```

Set `TM_TOKEN` for private Tasking Manager projects and `HF_TOKEN` (write) to publish, or put
them in `.env` for `just`.

## Build and publish a dataset

A dataset is one YAML config, such as those in [`configs/`](configs/) (run from a clone), or
your own file.

```bash
hot-oam-dataset --config configs/fair-guru-trees.yaml --all
hot-oam-dataset --config configs/fair-guru-trees.yaml --push-to-hub hotosm/fAIr-guru-td-trees
hot-oam-dataset --config configs/fair-guru-trees.yaml --update-root-catalog hotosm/fAIr-guru-stac
```

1. `--all` downloads tiles and labels, then writes the splits, Parquet, dataset card and STAC.
2. `--push-to-hub` uploads them and tags the release `v<dataset_version>`.
3. `--update-root-catalog` links that release in the root STAC catalogue,
   `https://huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json`
   ([open in STAC Browser](https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json)).

Raise `dataset_version` before `--all` for each new release. `--push-to-hub` stops when the
card was built for another version, or when that version is already on the Hub.

## Config

```yaml
output_dir: fAIr-guru-td-trees
dataset_version: 0.1.0
project_ids: 50661
feature_type: natural
tag_values: [tree, tree_group]
geometry_types: [point, polygon]
zoom: 20
burn_binary_masks: false
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
  summary: Tree training data from Freetown, Sierra Leone.
  selection: Labels are OpenStreetMap features tagged natural=tree or natural=tree_group.
  limitations:
    - diameter_crown is a mapper estimate.
```

The [config reference](https://hotosm.github.io/fAIr-ds-builder/reference/) explains every key.

## Commands

Run `hot-oam-dataset --help` for every option. The
[reference](https://hotosm.github.io/fAIr-ds-builder/reference/) lists the commands, config
keys and build steps.

## Development

```bash
just setup
just lint
just test
just dataset configs/fair-guru-trees.yaml
```

## Data sources and licensing

- Imagery: OpenAerialMap, CC-BY 4.0, "contributors of Open Imagery Network" ([source](https://openaerialmap.org/about/))
- Labels: OpenStreetMap, ODbL 1.0, "OpenStreetMap contributors" ([source](https://www.openstreetmap.org/copyright))
- Project metadata: [HOT Tasking Manager API](https://tasking-manager-production-api.hotosm.org/api/docs)

Tooling: GPL-3.0-or-later.

## Disclosure

Code was written with help from Claude and reviewed by the maintainers.
