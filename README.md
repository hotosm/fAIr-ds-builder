# hot-oam-dataset

Segmentation and object detection dataset builder for humanitarian mapping. Discovers validated mapping projects from the [HOT Tasking Manager](https://tasks.hotosm.org/) that use [OpenAerialMap](https://openaerialmap.org/) imagery, then generates 256x256 image chips with matching OpenStreetMap labels. Buildings are the default, and `--feature-type` selects any other OSM tag key.

## Setup

Install the `hot-oam-dataset` command from the repository:

```bash
uv tool install git+https://github.com/hotosm/tm-oam-ds-builder
```

Or, to work on the code, clone the repository, run `uv sync`, and prefix the commands below
with `uv run`. `uv sync --extra fair` adds what [Register in fAIr](#register-in-fair) needs.

Private Tasking Manager projects need a TM session token. Put it in `.env` as
`TM_TOKEN=<token>`. The `just` recipes load `.env`; when calling the CLI directly, use
`uv run --env-file .env hot-oam-dataset ...`. Without a token, hand-picked private projects
fail with a message naming `TM_TOKEN`, and discovery skips private projects with a warning.

## Usage

### Dataset config files

`--config` reads option values from a YAML file. Keys are the long option names with
underscores, and flags given on the command line override the file. `project_ids` uses the
same comma-separated form as `--project-ids`. `false` is accepted for options with a `--no-`
form, such as `burn_binary_masks`. The `split` key sets the split described under
[Splits](#splits), and the `card` key holds the dataset card text described under
[Dataset card](#dataset-card).

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
```

[`configs/`](https://github.com/hotosm/tm-oam-ds-builder/tree/master/configs) in the
repository holds the files for the published datasets. `--all` builds one end to end: tiles,
splits, Parquet, dataset card and STAC.

```bash
hot-oam-dataset --config configs/fair-guru-trees.yaml --all
```

Publish it with the same config, so the dataset version matches the build:

```bash
hot-oam-dataset --config configs/fair-guru-trees.yaml --push-to-hub hotosm/fAIr-guru-td-trees
```

### Discover projects

List all validated OAM projects from the last 5 years, writing `projects_summary.json` and
`projects_map.geojson` to the output directory:

```bash
hot-oam-dataset --discover-only --limit 0 --verbose
```

### Build dataset (default: 5 projects)

```bash
hot-oam-dataset --output-dir dataset
```

By default this also generates binary masks in `project_{id}/masks/`.
`HOT_OAM_BURN_BINARY_MASKS=false` changes the default.

### Build dataset with binary raster masks

```bash
hot-oam-dataset --output-dir dataset --burn-binary-masks
```

This writes per-chip masks to `project_{id}/masks/` with `0` for background and `255` for the labelled feature.

To disable mask generation:

```bash
hot-oam-dataset --output-dir dataset --no-burn-binary-masks
```

### Choose the feature to label

`--feature-type` takes an OSM tag key and sets the TM mapping type searched, the OSM
geometry requested, and the wording in reports.

```bash
hot-oam-dataset --output-dir road-dataset --feature-type highway --label-buffer-meters 5
```

| `--feature-type` | TM mapping type | OSM geometry | Reported as |
|------------------|-----------------|--------------|-------------|
| `building` | BUILDINGS | polygon | buildings |
| `highway` | ROADS | line | roads |
| `waterway` | WATERWAYS | line | waterways |
| `landuse` | LAND_USE | polygon | landuse areas |
| any other OSM key | OTHER | point, line, polygon | `<key>` features |

Override either default with `--tm-mapping-type` and `--geometry-types`.

By default every value of the tag is labelled. `--tag-values` keeps only the listed values,
for example trees mapped as points and tree groups mapped as polygons:

```bash
hot-oam-dataset --project-ids 50661 --output-dir trees --feature-type natural --tag-values tree tree_group --geometry-types point polygon --no-burn-binary-masks
```

`--require-tags` keeps only features that also carry the listed keys, with any value. Each
label's class is the value of the `--feature-type` tag, so road surfaces are labelled with:

```bash
hot-oam-dataset --project-ids 50793 --output-dir road-surface --feature-type surface --require-tags highway --geometry-types line --tm-mapping-type ROADS --no-burn-binary-masks
```

Point features rasterize to single pixels, so a point dataset is usually built without masks
and used through its GeoJSON labels.

`--label-buffer-meters` widens geometries before masks are burned, since points and lines
otherwise rasterize to one pixel. At zoom 19 near the equator, 5 m gives a road mask about 30 px wide. GeoJSON labels
keep their original geometry.

`HOT_OAM_FEATURE_TYPE` sets the default.

### Build full dataset (all projects)

```bash
hot-oam-dataset --limit 0 --output-dir dataset
```

### Custom limit and time range

```bash
hot-oam-dataset --limit 20 --years-back 3 --output-dir dataset
```

### Build a hand-picked set of projects into a named folder

Skip discovery and build exactly the TM project IDs you want. Useful for region-specific
datasets you intend to publish under their own Hugging Face repo:

```bash
hot-oam-dataset --project-ids 40695 --output-dir fAIr-guru-td-buildings
```

Validation and OAM-imagery checks are downgraded to warnings when IDs are hand-picked.
Only projects with validated tasks are built.
`--years-back` and `--limit` are ignored when `--project-ids` is set.

Each output directory is self-contained, with its own project index, stats, splits,
and dataset card.

### Add projects to a dataset folder

Rerun the build against the same output directory with the extra IDs. Existing projects
are kept and the stats and project index are extended.

```bash
hot-oam-dataset --project-ids 50793 --output-dir fAIr-guru-td-buildings
hot-oam-dataset --config configs/fair-guru-buildings.yaml --generate-splits
hot-oam-dataset --config configs/fair-guru-buildings.yaml --generate-parquet
```

### Splits

`--generate-splits` divides each project's tiles into contiguous regions with k-means on the
tile centres, seeded with the project id, and writes `splits.json`. Each region goes whole to
one split. The config's `split` section sets the rule, and its defaults are shown above:

| Setting | Default | Effect |
|---------|---------|--------|
| `regions_per_project` | 10 | Regions per project. A project with fewer tiles goes to train |
| `validation_regions` | 1 | Regions per project held out for validation |
| `test_regions` | 1 | Regions per project held out for test |
| `buffer_meters` | 0 | Leaves out validation and train tiles within this distance of a test tile, and train tiles within it of a validation tile |
| `exclude_shared_features` | false | Leaves out tiles holding part of a feature that also lies in a higher split |

With the defaults every tile is kept. Tiles left out by a buffer or the shared-feature rule
stay in the raw files and are counted on the dataset card. The split is recomputed from all tiles, so tiles can move between splits
when the project set changes. The `split` and `card` sections apply when a command is given
the dataset's `--config`; without one, the split defaults are used and the existing card is
updated in place.

`--negative-ratio` caps empty tiles in the Parquet at the given number per labelled tile, per
project. By default every empty tile is kept. Raw files keep every tile, and the STAC
catalogue follows the Parquet.

Run `hot-oam-dataset --help` for all CLI options.

## Output

Each output directory holds one `project_{id}/` folder per Tasking Manager project with its
chips, masks and per-tile GeoJSON labels, plus the dataset-wide stats, splits, Parquet,
STAC catalogue and dataset card.

### dataset_stats.json

Generated after every build, contains aggregate metadata for the entire dataset (abridged):

```json
{
  "total_projects": 100,
  "total_tile_pairs": 50000,
  "tiles_with_features": 42000,
  "tiles_without_features": 8000,
  "total_label_features": 580000,
  "avg_features_per_tile": 11.6,
  "feature_type": "building",
  "tag_values": [],
  "label_noun": "buildings",
  "class_counts": {"yes": 520000, "house": 60000},
  "tile_size_pixels": 256,
  "zoom_level": 19,
  "total_size_gb": 12.5,
  "imagery_source": "OpenAerialMap",
  "label_source": "OpenStreetMap",
  "label_format": "GeoJSON",
  "countries": {"Mexico": 18, "Kenya": 12, "Myanmar": 5},
  "organisations": {"HOT": 40, "Missing Maps": 15}
}
```

`feature_type`, `zoom_level` and `tile_size_pixels` come from each project's `metadata.json`, and become a list
when a directory holds projects built with different settings.

## How it works

1. Queries HOT Tasking Manager API for published and archived projects
2. Filters for the mapping type implied by `--feature-type` with >= 95% validation
3. Filters for projects using OpenAerialMap URLs
4. Skips a project when a probe tile from its imagery URL fails
5. Downloads 256x256 tiles at the `--zoom` level, 19 by default, from OAM using [geoml-toolkits](https://github.com/kshitijrajsharma/geoml-toolkits), keeping only tiles that fall fully inside tasks marked `VALIDATED`
6. Downloads the OSM features intersecting those tiles via the [HOT Raw Data API](https://github.com/hotosm/raw-data-api)
7. Splits the features into per-tile GeoJSON labels
8. Validates chip-label pairs, removes corrupt chips, chips where the imagery server marked more than 1% of pixels as missing (transparent, usually outside the survey), and orphan labels
9. Generates per-project and aggregate dataset statistics

### Missing imagery threshold

Imagery servers return transparent pixels where a survey has no data, and also mark some
pure black pixels inside the imagery, typically deep shadows, as transparent. A chip is
removed when more than 1% of its pixels are transparent (`MAX_MISSING_IMAGERY_SHARE`).
Measured on 13,302 chips with any transparent pixel:

| Transparent share | Chips | Share in one region touching the tile edge |
|-------------------|-------|-------------------------------------------|
| up to 0.1%        | 12,594 | 2% |
| 0.1% to 1%        | 326    | 8% |
| 1% to 5%          | 56     | 66% |
| above 5%          | 326    | 100% |

Below 1% the transparent pixels are mostly small scattered specks inside the image; above it
they are mostly gaps along the survey edge.

## Publishing to Hugging Face

Set repo once for both push commands:

```bash
export HF_DATASET_REPO_ID=hotosm/fAIr-guru-td-buildings
```

### Login

```bash
uv run python -c "from huggingface_hub import login; login()"
```

Or set `HF_TOKEN` environment variable (write-access token from huggingface.co/settings/tokens):

```bash
export HF_TOKEN=hf_your_token_here
```

### Generate Parquet (for dataset preview)

```bash
hot-oam-dataset --generate-parquet --output-dir dataset
```

This writes `parquet/train.parquet`, `parquet/validation.parquet` and `parquet/test.parquet` when `splits.json` exists, otherwise `parquet/data.parquet`. Rows hold labels, tile coordinates and project metadata, and reference chips and masks by their local path under `--output-dir`. `--push-to-hub` embeds the image bytes in the uploaded Parquet.

### Push to Hugging Face Hub

```bash
hot-oam-dataset --push-to-hub hotosm/fAIr-guru-td-buildings --output-dir dataset
```

With env repo id:

```bash
hot-oam-dataset --push-to-hub --output-dir dataset
```

This will:

1. Generate Parquet when `parquet/` is empty. An existing Parquet is pushed as is.
2. Push the dataset (Parquet with embedded images) to the Hugging Face Hub
3. Upload the dataset's card as `README.md`, and `samples.png` when it exists
4. Upload `stac/` when it exists
5. Tag the revision `v<dataset-version>`

### Dataset card

The card layout ships with the package (`src/hot_oam_dataset/dataset_card.md`), and each
dataset's own text comes from its config. `--generate-parquet` writes
`<output-dir>/dataset_card.md`, with the front matter, split sizes and a generated block
counted from the rows written to the Parquet: a summary table of tiles, splits, labels, pixel
size and area, a grid of sample tiles beside their labels, the class table with the
OpenStreetMap wiki pages of the tags, and the source projects.

A config's `card` section holds the text written by hand for one dataset:

```yaml
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
```

All seven keys are required. An optional `sample_tiles` sets how many sample tiles the card
shows, 6 by default and 0 to leave the grid out; they are drawn to `<output-dir>/samples.png`
and uploaded with the card. `tasks` lists one or more of `segmentation`, `detection` and
`classification`, the STAC label extension's task names; it sets the card's task categories and
the `label:tasks` of the STAC items. `repo_id` fills the load and QGIS examples, and `label_noun` names the labels in
the card tables and the STAC description. With a `card` section the card is rewritten from
the template on every `--generate-parquet`: the title and summary open the card, `selection`
follows the explanation of how Tasking Manager projects are mapped, and the limitations get
their own section. A dataset built without a config copies the template on first use, and
later runs update only its stats and front matter, so edits to the copy are kept.

`--push-to-hub` stops with an error when the card's dataset version differs from
`--dataset-version`; rerun `--generate-parquet` with that version. It also stops when the
card has no generated stats yet, which includes cards generated by earlier releases of this
package.

### Push raw files (optional)

Upload each `project_{id}/` folder (chips, masks, labels, metadata) under `raw/`, and
`dataset_stats.json`, `projects_summary.json` and `projects_map.geojson` at the repo root.
`--compress-raw` uploads one `data.tar.gz` per project in place of the chip, mask and label
folders. An existing archive is reused as is.

```bash
hot-oam-dataset --push-raw hotosm/fAIr-guru-td-buildings --output-dir dataset
```

With env repo id:

```bash
hot-oam-dataset --push-raw --output-dir dataset
```

The chips are the tiles as served, each with a GDAL `.aux.xml` sidecar holding its georeference, so GIS tools place them on the map. The Parquet push is sufficient for ML use.

### Updating the dataset

Run the build again with `--all`, which also regenerates the splits, Parquet and STAC, then
push with a new `--dataset-version`. Published links point at the `v<dataset-version>` tag, so
`--push-to-hub` stops when that version was already pushed:

```bash
hot-oam-dataset --limit 0 --output-dir dataset --all --dataset-version 0.4.0
hot-oam-dataset --push-to-hub hotosm/fAIr-guru-td-buildings --output-dir dataset --dataset-version 0.4.0
hot-oam-dataset --push-raw hotosm/fAIr-guru-td-buildings --output-dir dataset --dataset-version 0.4.0
```

## STAC catalogue and QGIS

```bash
hot-oam-dataset --config configs/fair-guru-trees.yaml --generate-stac
```

This reads the split Parquet files and writes `stac/`, a static STAC Collection. It needs the
split Parquet files and `dataset_stats.json`, so run `--generate-splits` and `--generate-parquet`
first, or use `--all`. Each Tasking Manager project has two items, laid out as the
[STAC label extension](https://github.com/stac-extensions/label) v1.0.1 describes:

| Item | Assets | Describes |
|------|--------|-----------|
| `project-<id>` | `labels` (GeoJSON), `labels_parquet` (GeoParquet), `tiles` (GeoParquet) | The labels: `label:type`, `label:tasks`, `label:classes`, `label:properties`, `label:methods`, `label:overviews` with the count per class, a `source` link to the imagery item, a `via` link to the Tasking Manager project, the project under `tm:*`, and the tile counts per split under `dataset:*` |
| `project-<id>-imagery` | `imagery` (Cloud Optimized GeoTIFF, JPEG) | The chips placed at their tile positions in EPSG:3857, with the ground pixel size in metres at the project's centre latitude as `raster:spatial_resolution` ([raster](https://github.com/stac-extensions/raster) v2.0.0), the capture date OpenAerialMap reports for the source image (from its [catalogue](https://api.openaerialmap.org/meta), or the [HOT imagery API](https://api.imagery.hotosm.org/stac) for imagery served there), and an `xyz` link to the OpenAerialMap tiles ([web-map-links](https://github.com/stac-extensions/web-map-links) v1.3.0) |

The labels hold one row per label as clipped to its tile, with `class`, OSM tags, `tile_id` and
`split`; the tiles hold each tile footprint with `split`, `num_features` and `class_counts`. The
Collection summarises the tasks, classes and zoom levels, and links the
OpenStreetMap wiki pages of the tags the labels were selected on as `describedby`; the card
lists the same pages. `--generate-stac` looks up each project's imagery in the OpenAerialMap
catalogue, so it needs network access. The imagery mosaic is a viewing copy,
and the Parquet splits keep the chips as served. `stac/split_map.png` shows the tiles of each
project coloured by split and is embedded in the dataset card. `--push-to-hub` uploads a copy
of `stac/` to the same repo, with every link absolute under
`https://huggingface.co/datasets/<repo>/resolve/v<dataset-version>/stac`; the local `stac/`
keeps relative links, so it can be moved or opened from disk. The Collection license is `ODbL-1.0`, and its license links name CC BY 4.0 for
the imagery and ODbL 1.0 for the labels. Each part keeps its own license, as listed under Data
Sources & Licensing.

To browse it in QGIS, right-click STAC in the Browser panel, choose New STAC Connection, and
enter the URL of `collection.json`, for a published dataset
`https://huggingface.co/datasets/<repo>/resolve/main/stac/collection.json`. Dragging an imagery
item onto the map adds the imagery. From version 4.0, dragging a labels item adds its GeoParquet
labels and tiles; QGIS adds STAC assets of the COG, COPC, EPT, Zarr and Parquet formats and
skips GeoJSON ([source](https://github.com/qgis/QGIS/blob/release-4_0/src/core/stac/qgsstacasset.cpp)).
QGIS 3.44 adds only COG, COPC and EPT assets
([source](https://github.com/qgis/QGIS/blob/release-3_44/src/core/stac/qgsstacasset.cpp)), so
there the labels open with Layer, Add Layer, Add Vector Layer, from the asset URL.

## Register in fAIr

`--publish-stac` registers a published dataset in a fAIr STAC API, such as
`https://stac.ai.hotosm.org/stac`, as one item in its `datasets` collection. The item is built
and written with [fair-py-ops](https://github.com/hotosm/fAIr-models), installed with the
`fair` extra (`uv tool install "hot-oam-dataset[fair] @ git+https://github.com/hotosm/tm-oam-ds-builder"`).
It needs the config's `card` section and a `fair` section naming who registers the dataset:

```yaml
fair:
  user_id: hot
  provider: hot
```

```bash
export FAIR_STAC_API_KEY=<key>
hot-oam-dataset --config configs/fair-guru-trees.yaml --publish-stac https://stac.ai.hotosm.org/stac
```

Run it after `--push-to-hub` with the same version: it needs the Hub repo's
`v<dataset-version>` tag. The item's `chips` and `labels` assets point at that tag's Parquet
(`hf://datasets/<repo>@v<version>/data`), so the item keeps pointing at the data it describes
when the dataset is updated.

## Just Recipes

Requires [just](https://github.com/casey/just). Run `just` to list all available recipes.

- `just lint` runs ruff fix and format, ty on `src/`, the pre-commit hooks and pytest.
- `just dataset configs/fair-guru-trees.yaml` builds one dataset from its config, the same as `--all`.
- `just build --limit 0 --years-back 3` builds into `dataset/` with any extra CLI flags, then writes a `data.tar.gz` per project.
- `just build-projects "40695,50793" fAIr-guru-td-buildings` builds a hand-picked set into a named folder, then writes a `data.tar.gz` per project.
- `just publish hotosm/fAIr-guru-td-buildings fAIr-guru-td-buildings` generates the Parquet and pushes it to the Hub. A third argument `raw` also uploads the raw files as per-project archives. It regenerates the Parquet without a config, so use it only for directories built without one.
- `just publish-projects fAIr-guru-td-buildings hotosm/fAIr-guru-td-buildings raw` publishes a named folder, with the same optional `raw`.

## Data Sources & Licensing

- **Imagery:** [OpenAerialMap](https://openaerialmap.org/): CC-BY 4.0, attribution to "contributors of Open Imagery Network" ([source](https://openaerialmap.org/about/))
- **Labels:** [OpenStreetMap](https://www.openstreetmap.org/): ODbL 1.0, attribution to "OpenStreetMap contributors" ([source](https://www.openstreetmap.org/copyright))
- **Project metadata:** [HOT Tasking Manager API](https://tasking-manager-production-api.hotosm.org/api/docs)

## License

Tooling: GPL-3.0-or-later. Dataset imagery: CC-BY 4.0. Dataset labels: ODbL 1.0.

## Disclosure

LLM Claude was used to assist writing code, and the result was reviewed by the maintainers.
