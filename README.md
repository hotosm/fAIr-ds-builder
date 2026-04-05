# hot-oam-dataset

Semantic segmentation dataset builder for building disaster mapping. Discovers validated building mapping projects from the [HOT Tasking Manager](https://tasks.hotosm.org/) that use [OpenAerialMap](https://openaerialmap.org/) imagery, then generates 256x256 image chips with corresponding building footprint labels from OpenStreetMap.

## Setup

```bash
uv sync
```

## Usage

### Discover projects (no download)

List all validated OAM building projects from the last 5 years:

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

This writes per-chip masks to `project_{id}/masks/` with `0` for background and `255` for buildings.

To disable mask generation:

```bash
uv run hot-oam-dataset --output-dir dataset --no-burn-binary-masks
```

### Configure label feature type (building, highway, etc.)

```bash
export HOT_OAM_FEATURE_TYPE=building
uv run hot-oam-dataset --output-dir dataset
```

Or pass it directly:

```bash
uv run hot-oam-dataset --output-dir dataset --feature-type highway
```

### Build full dataset (all projects)

```bash
uv run hot-oam-dataset --limit 0 --output-dir dataset
```

### Custom limit and time range

```bash
uv run hot-oam-dataset --limit 20 --years-back 3 --output-dir dataset
```

Run `uv run hot-oam-dataset --help` for all CLI options.

## Output Structure

```text
dataset/
  dataset_stats.json               # Aggregate stats for the whole dataset
  projects_summary.json            # All discovered projects
  projects_map.geojson             # GeoJSON map of project areas
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
      OAM-{x}-{y}-{z}.geojson     # Per-tile building footprint polygons
      osm-result.geojson           # Full OSM building data for the area
```

### dataset_stats.json

Generated after every build, contains aggregate metadata for the entire dataset:

```json
{
  "total_projects": 100,
  "total_tile_pairs": 50000,
  "tiles_with_buildings": 42000,
  "tiles_without_buildings": 8000,
  "total_building_polygons": 580000,
  "avg_buildings_per_tile": 11.6,
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

## How it works

1. Queries HOT Tasking Manager API for projects with custom imagery
2. Filters for projects using OpenAerialMap URLs
3. Filters for building mapping type with >= 95% validation
4. Downloads 256x256 tiles at zoom 19 from OAM using [geoml-toolkits](https://github.com/kshitijrajsharma/geoml-toolkits)
5. Downloads OSM building footprints via HOT Raw Data API
6. Splits building polygons into per-tile GeoJSON labels
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

This generates a `parquet/data.parquet` file with embedded images and masks, labels, tile coordinates, and project metadata. HuggingFace uses this for the dataset viewer/preview.

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
3. Upload `dataset_card.md` as `README.md` to the repo (dataset card / model card)

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
just lint                                         # ruff lint + format + ty + pytest
just build --limit 0 --years-back 3               # pass any CLI flags through
just publish hotosm/vhr-building-segmentation raw   # parquet + push to HF Hub (+ raw)
```

## Data Sources & Licensing

- **Imagery:** [OpenAerialMap](https://openaerialmap.org/) - CC-BY 4.0, attribution to "contributors of Open Imagery Network"
- **Labels:** [OpenStreetMap](https://www.openstreetmap.org/) - ODbL 1.0, attribution to "OpenStreetMap contributors"
- **Project metadata:** [HOT Tasking Manager API](https://tasking-manager-production-api.hotosm.org/api/docs)

## License

Tooling: GPL-3.0-or-later. Dataset imagery: CC-BY 4.0. Dataset labels: ODbL 1.0.

## Disclosure

LLM Claude was used to assist writing code and carefully reviwed !