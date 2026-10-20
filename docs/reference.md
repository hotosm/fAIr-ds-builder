# Reference

## Commands

| Option | Does |
|--------|------|
| `--all` | Build, then write splits, Parquet, card and STAC |
| `--discover-only` | List validated OpenAerialMap projects |
| `--generate-splits`, `--generate-parquet`, `--generate-stac` | Run one step of `--all`, in this order |
| `--push-to-hub REPO` | Publish the release and tag it `v<dataset_version>` |
| `--push-raw REPO`, `--compress-raw` | Upload the per-project chips, masks and labels |
| `--update-root-catalog REPO` | Link the release in the root STAC catalogue |
| `--publish-stac URL` | Register the release in a fAIr STAC API |

## Config keys

Top-level keys are the command options with underscores. Command-line flags override them.

| Key | Effect |
|-----|--------|
| `output_dir` | Folder the dataset is built in |
| `dataset_version` | Version written to the card, the Parquet and the Hub tag |
| `project_ids` | Tasking Manager projects to build. When unset, validated OpenAerialMap projects are discovered |
| `feature_type` | OSM tag key to label. Its value is each label's class |
| `tag_values` | Keep only these values of the tag |
| `require_tags` | Keep only features that also carry these keys, such as `highway` for road surfaces |
| `geometry_types` | `point`, `line`, `polygon` or `all_geometry` |
| `tm_mapping_type` | Tasking Manager mapping type the projects are checked against |
| `zoom` | Tile zoom level |
| `burn_binary_masks` | Write a binary mask per tile (0 background, 255 feature) |
| `label_buffer_meters` | Widen labels by this many metres before masks are burned, so points and lines show |
| `negative_ratio` | Keep at most this many empty tiles per labelled tile, per project, in the Parquet |
| `split` | How tiles are split, see below |
| `card` | Hand-written dataset card text, see below |
| `fair` | `user_id` and `provider` for `--publish-stac` |

### Feature types

| `feature_type` | Mapping type | Geometry |
|----------------|--------------|----------|
| `building` | BUILDINGS | polygon |
| `highway` | ROADS | line |
| `waterway` | WATERWAYS | line |
| `landuse` | LAND_USE | polygon |
| any other key | OTHER | point, line, polygon |

### `split`

Each project's tiles are grouped into contiguous regions with k-means on the tile centres,
seeded with the project id. Each region goes whole to one split.

| Setting | Default | Effect |
|---------|---------|--------|
| `regions_per_project` | 10 | Regions per project. A project with fewer tiles goes to train |
| `validation_regions` | 1 | Regions held out for validation |
| `test_regions` | 1 | Regions held out for test |
| `buffer_meters` | 0 | Drop train and validation tiles within this distance of a higher split (test, then validation) |
| `exclude_shared_features` | false | Leave out tiles holding part of a feature that also lies in a higher split |

Left-out tiles stay in the raw files and are counted on the card.

### `card`

| Key | Effect |
|-----|--------|
| `repo_id` | Hub repo of the dataset, used in the card's links and by `--update-root-catalog` and `--publish-stac` |
| `title`, `summary` | Open the card |
| `label_noun` | Names the labels in the card tables, such as `trees` |
| `tasks` | `segmentation`, `detection` or `classification`; also the STAC `label:tasks` |
| `selection` | How the labels were chosen |
| `limitations` | List shown under Known limitations |
| `sample_tiles` | Optional, sample tiles shown on the card (default 6, 0 for none) |

All keys except `sample_tiles` are required.

`--generate-parquet` fills the card template with counts, sample tiles, classes and source
projects from the Parquet rows.

## Environment variables

| Variable | Effect |
|----------|--------|
| `TM_TOKEN` | Tasking Manager session token for private projects |
| `HF_TOKEN` | Hugging Face write token, read by [huggingface_hub](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken) |
| `HF_DATASET_REPO_ID` | Repo for `--push-to-hub` and `--push-raw` when none is given |
| `HOT_OAM_FEATURE_TYPE` | Default `--feature-type` |
| `HOT_OAM_BURN_BINARY_MASKS` | Default for `--burn-binary-masks` |
| `FAIR_STAC_API_KEY` | Key for `--publish-stac` |

## How a build works

1. Find Tasking Manager projects of the mapping type, at least 95% validated, with OpenAerialMap imagery.
2. Download 256x256 tiles that lie fully inside tasks marked `VALIDATED`, with [geoml-toolkits](https://github.com/kshitijrajsharma/geoml-toolkits).
3. Download the OSM features on those tiles from the [HOT Raw Data API](https://github.com/hotosm/raw-data-api) and clip them per tile.
4. Drop corrupt chips (the tile images), orphan labels, and chips with more than 1% transparent pixels (gaps at the survey edge).

## Output folder

| Path | Content |
|------|---------|
| `project_<id>/` | Chips, masks, per-tile GeoJSON labels and `metadata.json` |
| `projects_summary.json`, `projects_map.geojson` | Discovered projects |
| `dataset_stats.json` | Tile, label and class counts, in total and per project |
| `splits.json` | Split of every tile |
| `parquet/` | `train`, `validation` and `test` Parquet |
| `dataset_card.md`, `samples.png` | Dataset card and its sample tiles |
| `stac/` | STAC Collection, see below |

## STAC

`--generate-stac` writes a static STAC Collection with one item, `project-<id>`, per Tasking
Manager project, following the [label extension](https://github.com/stac-extensions/label)
v1.0.1. The item carries the classes, counts per class and split, and the Tasking Manager
project. Its assets:

| Asset | Content |
|-------|---------|
| `imagery` | Cloud Optimized GeoTIFF of the dataset tiles, with the capture date and `raster:spatial_resolution` |
| `labels`, `labels_parquet` | Labels with OSM tags and split (GeoJSON, GeoParquet) |
| `tiles` | Tile footprints with split (GeoParquet) |

The local `stac/` uses relative links. `--push-to-hub` uploads a copy with absolute links to
the release tag: STAC JSON under `https://huggingface.co/datasets/<repo>/raw/v<version>/stac/`
for QGIS, and assets under `.../resolve/v<version>/stac/`.

`--update-root-catalog <root-repo>` adds the release's Collection as a `child` of
`catalog.json` in the root repo, replacing any other release of the same dataset. The first
run creates the repo.

## fAIr

`--publish-stac <url>` writes one item to the `datasets` collection of a fAIr STAC API with
[fair-py-ops](https://github.com/hotosm/fAIr-models). It runs after `--push-to-hub` and needs the
`fair` extra, `FAIR_STAC_API_KEY` and the config's `card` and `fair` sections. The item's assets
point at `hf://datasets/<repo>@v<version>/data`.
