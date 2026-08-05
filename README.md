# hot-oam-dataset

Semantic segmentation dataset builder for humanitarian mapping. Discovers validated mapping projects from the [HOT Tasking Manager](https://tasks.hotosm.org/) that use [OpenAerialMap](https://openaerialmap.org/) imagery, then generates 256x256 image chips with matching OpenStreetMap labels. Buildings are the default; any OSM feature type can be labelled instead.

## Setup

```bash
uv sync
```

## Usage

### Discover projects (no download)

List all validated OAM projects from the last 5 years:

```bash
uv run hot-oam-dataset --discover-only --verbose
```

### Build dataset (default - 5 projects)

```bash
uv run hot-oam-dataset --output-dir dataset
```

By default this also generates binary masks in `project_{id}/masks/`.

### Build dataset with binary raster masks

```bash
uv run hot-oam-dataset --output-dir dataset --burn-binary-masks
```

This writes per-chip masks to `project_{id}/masks/` with `0` for background and `255` for the labelled feature.

To disable mask generation:

```bash
uv run hot-oam-dataset --output-dir dataset --no-burn-binary-masks
```

### Choose the feature to label

`--feature-type` takes an OSM tag key and sets the TM mapping type searched, the OSM
geometry requested, and the wording in reports.

```bash
uv run hot-oam-dataset --output-dir road-dataset --feature-type highway --label-buffer-meters 5
```

| `--feature-type` | TM mapping type | OSM geometry | Reported as |
|------------------|-----------------|--------------|-------------|
| `building` | BUILDINGS | polygon | buildings |
| `highway` | ROADS | line | roads |
| `waterway` | WATERWAYS | line | waterways |
| `landuse` | LAND_USE | polygon | landuse areas |
| any other OSM key | OTHER | point, line, polygon | `<key>` features |

Override either default with `--tm-mapping-type` and `--geometry-types`.

`--label-buffer-meters` widens geometries before masks are burned, since lines otherwise
rasterize to one pixel. At zoom 19, 5 m gives a road mask about 30 px wide. GeoJSON labels
keep their original geometry.

`HOT_OAM_FEATURE_TYPE` sets the default.

### Build full dataset (all projects)

```bash
uv run hot-oam-dataset --limit 0 --output-dir dataset
```

### Custom limit and time range

```bash
uv run hot-oam-dataset --limit 20 --years-back 3 --output-dir dataset
```

### Build a hand-picked set of projects into a named folder

Skip discovery and build exactly the TM project IDs you want. Useful for region-specific
datasets you intend to publish under their own HuggingFace repo:

```bash
uv run hot-oam-dataset --project-ids 25317 --output-dir freetown-building-dataset
```

Validation and OAM-imagery checks are downgraded to warnings when IDs are hand-picked.
`--years-back` and `--limit` are ignored when `--project-ids` is set.

Each output directory is self-contained, with its own project index, stats, splits,
and dataset card.

### Add projects to a dataset folder

Rerun the build against the same output directory with the extra IDs. Existing projects
are kept and the stats and project index are extended.

```bash
uv run hot-oam-dataset --project-ids 25320 --output-dir freetown-building-dataset
uv run hot-oam-dataset --generate-splits  --output-dir freetown-building-dataset
uv run hot-oam-dataset --generate-parquet --output-dir freetown-building-dataset
```

`--generate-splits` recomputes all assignments, so projects can move between splits when
the project set changes.

Run `uv run hot-oam-dataset --help` for all CLI options.

## Output Structure

```text
dataset/
  dataset_stats.json               # Aggregate stats for the whole dataset
  projects_summary.json            # All discovered projects
  projects_map.geojson             # GeoJSON map of project areas
  splits.json                      # Project to train/validation/test assignments
  dataset_card.md                  # This dataset's card, published as the repo README
  parquet/
    data.parquet                   # HuggingFace-compatible Parquet with embedded images
  project_{id}/
    metadata.json                  # Project metadata + per-project stats
    aoi.geojson                    # Area of interest
    tiles.geojson                  # Tile boundary geometries
    chips/
      OAM-{x}-{y}-{z}.tif         # 256x256 aerial imagery tiles
    masks/
      OAM-{x}-{y}-{z}.tif         # Binary masks (0 background, 255 feature)
    labels/
      OAM-{x}-{y}-{z}.geojson     # Per-tile label geometries
      osm-result.geojson           # Full OSM label data for the area
```

### dataset_stats.json

Generated after every build, contains aggregate metadata for the entire dataset:

```json
{
  "total_projects": 100,
  "total_tile_pairs": 50000,
  "tiles_with_features": 42000,
  "tiles_without_features": 8000,
  "total_label_features": 580000,
  "avg_features_per_tile": 11.6,
  "feature_type": "building",
  "label_noun": "buildings",
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

`feature_type` and `zoom_level` come from each project's `metadata.json`, and become a list
when a directory holds projects built with different settings.

## How it works

1. Queries HOT Tasking Manager API for projects with custom imagery
2. Filters for projects using OpenAerialMap URLs
3. Filters for the mapping type implied by `--feature-type` with >= 95% validation
4. Downloads 256x256 tiles at zoom 19 from OAM using [geoml-toolkits](https://github.com/kshitijrajsharma/geoml-toolkits)
5. Downloads the matching OSM features via HOT Raw Data API
6. Splits the features into per-tile GeoJSON labels
7. Validates chip-label pairs, removes corrupt chips and orphan labels
8. Generates per-project and aggregate dataset statistics

## Publishing to Hugging Face

Set repo once for both push commands:

```bash
export HF_DATASET_REPO_ID=hotosm/vhr-building-segmentation
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
uv run hot-oam-dataset --generate-parquet --output-dir dataset
```

This generates a `parquet/data.parquet` file with embedded images and masks, labels, tile coordinates, and project metadata. Each row carries `feature_type` and `num_features`. HuggingFace uses this for the dataset viewer/preview.

### Push to HuggingFace Hub

```bash
uv run hot-oam-dataset --push-to-hub hotosm/vhr-building-segmentation --output-dir dataset
```

With env repo id:

```bash
uv run hot-oam-dataset --push-to-hub --output-dir dataset
```

This will:

1. Generate Parquet if not already present
2. Push the dataset (parquet with embedded images) to HuggingFace Hub
3. Upload the dataset's card as `README.md` to the repo (dataset card / model card)

### Dataset card

Root `dataset_card.md` is the template. It is copied to `<output-dir>/dataset_card.md` on
first use, and that copy receives the generated stats and split sizes and gets published.

Edit the copy to describe one dataset, the template to change the starting point for new ones.

### Push raw files (optional)

Upload the full raw dataset (chips, labels, metadata) under `raw/` in the same repo:

```bash
uv run hot-oam-dataset --push-raw hotosm/vhr-building-segmentation --output-dir dataset
```

With env repo id:

```bash
uv run hot-oam-dataset --push-raw --output-dir dataset
```

Useful for GIS users who need georeferenced GeoTIFFs or raw GeoJSON labels. The Parquet push is sufficient for ML use.

### Updating the dataset

Run the build again, then re-push:

```bash
uv run hot-oam-dataset --limit 0 --output-dir dataset
uv run hot-oam-dataset --push-to-hub hotosm/vhr-building-segmentation --output-dir dataset
uv run hot-oam-dataset --push-raw hotosm/vhr-building-segmentation --output-dir dataset
```

## Quick visualization

- Open `projects_map.geojson` in QGIS.
- Each project feature includes imagery/TMS URL in properties (`imagery_url`) so you can quickly visualize vector project footprints with source raster links.

## Just Recipes

Requires [just](https://github.com/casey/just). Run `just` to list all available recipes.

```bash
just lint                                                                # ruff lint + format + ty + pytest
just build --limit 0 --years-back 3                                      # pass any CLI flags through
just publish hotosm/vhr-building-segmentation                            # parquet + push to HF Hub
just publish hotosm/vhr-building-segmentation dataset raw                # also upload raw files
just build-projects "25317" freetown-building-dataset                    # build a hand-picked set into a named folder
just publish-projects freetown-building-dataset hotosm/freetown-buildings raw   # publish that folder (+ raw)
```

## Data Sources & Licensing

- **Imagery:** [OpenAerialMap](https://openaerialmap.org/) - CC-BY 4.0, attribution to "contributors of Open Imagery Network"
- **Labels:** [OpenStreetMap](https://www.openstreetmap.org/) - ODbL 1.0, attribution to "OpenStreetMap contributors"
- **Project metadata:** [HOT Tasking Manager API](https://tasking-manager-production-api.hotosm.org/api/docs)

## License

Tooling: GPL-3.0-or-later. Dataset imagery: CC-BY 4.0. Dataset labels: ODbL 1.0.

## Disclosure

LLM Claude was used to assist writing code, and the result was carefully reviewed.
