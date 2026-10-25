# Build a dataset

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/hotosm/fAIr-ds-builder
```

Set `TM_TOKEN` (your Tasking Manager session token) for private projects and `HF_TOKEN` (a
Hugging Face write token) to publish.

## Write a config

Copy one from [`configs/`](https://github.com/hotosm/fAIr-ds-builder/tree/master/configs)
and change the projects, feature type and card text. The [Reference](../reference.md) lists
every key.

## Build

```bash
hot-oam-dataset --config my-dataset.yaml --all
```

Writes tiles, splits, Parquet, card and STAC to `output_dir`.

## Publish

```bash
hot-oam-dataset --config my-dataset.yaml --push-to-hub <your-hf-user>/my-dataset
hot-oam-dataset --config my-dataset.yaml --update-root-catalog <your-hf-user>/my-stac
```

`--update-root-catalog` adds the release to a STAC catalogue repo you can write to. HOT
maintainers use `hotosm/fAIr-guru-stac`.

The push tags the release `v<dataset_version>`. Raise `dataset_version` before `--all` for
each new release: the push stops when the card was built for another version, or when that
version is already on the Hub.

## Register in fAIr

```bash
uv tool install "hot-oam-dataset[fair] @ git+https://github.com/hotosm/fAIr-ds-builder"
export FAIR_STAC_API_KEY=<key>
hot-oam-dataset --config my-dataset.yaml --publish-stac https://stac.ai.hotosm.org/stac
```
