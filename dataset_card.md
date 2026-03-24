---
language:
  - en
license:
  - cc-by-4.0
  - odbl
size_categories:
  - 10K<n<100K
task_categories:
  - image-segmentation
task_ids:
  - semantic-segmentation
tags:
  - buildings
  - disaster-mapping
  - remote-sensing
  - satellite-imagery
  - openstreetmap
  - openaerialmap
  - humanitarian
  - hot-tasking-manager
  - geospatial
  - segmentation
pretty_name: HOT Building Segmentation Dataset
dataset_info:
  features:
    - name: image
      dtype: image
    - name: tile_id
      dtype: string
    - name: tile_x
      dtype: int32
    - name: tile_y
      dtype: int32
    - name: tile_z
      dtype: int32
    - name: project_id
      dtype: int32
    - name: project_name
      dtype: string
    - name: country
      dtype: string
    - name: organisation
      dtype: string
    - name: imagery_url
      dtype: string
    - name: num_buildings
      dtype: int32
    - name: label_geojson
      dtype: large_string
    - name: bbox_west
      dtype: float64
    - name: bbox_south
      dtype: float64
    - name: bbox_east
      dtype: float64
    - name: bbox_north
      dtype: float64
  splits:
    - name: train
      num_examples: 0
---

# HOT Building Segmentation Dataset

## Dataset Description

A semantic segmentation dataset for building footprint extraction from aerial imagery, built from validated [Humanitarian OpenStreetMap Team (HOT)](https://www.hotosm.org/) Tasking Manager projects that use [OpenAerialMap (OAM)](https://openaerialmap.org/) imagery.

### Dataset Summary

This dataset pairs 256x256 aerial image tiles (zoom level 19) from OpenAerialMap with building footprint labels from OpenStreetMap. All source projects have been fully validated through the HOT Tasking Manager, ensuring high label quality from expert humanitarian mappers.

**Target use case:** Training and evaluating deep learning models for building detection and segmentation in disaster mapping contexts.

### Supported Tasks

- **Semantic Segmentation:** Pixel-level building vs. background classification
- **Instance Segmentation:** Individual building footprint delineation (using GeoJSON polygon labels)
- **Object Detection:** Building bounding box detection (derivable from polygon labels)

### Languages

English (metadata and documentation)

## Dataset Structure

### Data Format

```
dataset/
  project_{id}/
    metadata.json
    aoi.geojson
    tiles.geojson                  # Tile boundary geometries
    chips/
      OAM-{x}-{y}-{z}.tif         # 256x256 aerial imagery tiles
    labels/
      OAM-{x}-{y}-{z}.geojson     # Per-tile building footprint polygons
      osm-result.geojson           # Full OSM building data for the area
  parquet/
    data.parquet                   # HuggingFace Parquet with embedded images
  projects_summary.json
  projects_map.geojson
```

### Data Fields

**Image tiles (chips):**
- Format: GeoTIFF (.tif), georeferenced
- Size: 256x256 pixels
- Zoom level: 19 (~0.3m/pixel at equator)
- Source: OpenAerialMap drone/aerial imagery
- Naming: `OAM-{x}-{y}-{z}.tif` following standard web map tile coordinates

**Labels (GeoJSON):**
- Format: GeoJSON with building footprint polygons
- Source: OpenStreetMap building data via HOT Raw Data API
- Coordinate system: EPSG:4326 (WGS84)
- Each file corresponds to one image tile with matching filename

**Metadata:**
- `metadata.json`: Project-level information (TM project ID, name, imagery URL, country, validation status)
- `aoi.geojson`: Project area of interest boundary
- `projects_summary.json`: Summary of all included projects
- `projects_map.geojson`: Map of all project areas

### Data Splits

The dataset is provided unsplit. Users should create their own train/validation/test splits based on their requirements. Recommended approach: split by project to avoid spatial autocorrelation between splits.

## Dataset Creation

### Source Data

**Imagery:** OpenAerialMap (OAM), a repository of openly licensed aerial imagery collected by drones, balloons, and satellites. Licensed under CC-BY or similar open licenses.

**Labels:** OpenStreetMap (OSM) building footprints, contributed by humanitarian mappers through HOT Tasking Manager projects. Licensed under ODbL 1.0.

**Project Selection Criteria:**
- Uses OpenAerialMap imagery (custom TMS URL containing `openaerialmap.org`)
- Mapping type includes BUILDINGS
- Validation completion >= 95%
- Created within the specified time window (default: last 5 years)

### Data Collection Process

1. **Project Discovery:** Query HOT Tasking Manager API for projects with custom imagery, filter for OAM URLs and building mapping type
2. **Quality Filter:** Retain only projects with >= 95% validation completion
3. **Tile Generation:** Generate 256x256 tiles at zoom level 19 within each project's area of interest
4. **Imagery Download:** Fetch aerial imagery tiles from OpenAerialMap TMS endpoints
5. **Label Download:** Fetch building footprints from OpenStreetMap via HOT Raw Data API
6. **Label Splitting:** Clip building polygons to individual tile boundaries

Tools used: [geoml-toolkits](https://github.com/kshitijrajsharma/geoml-toolkits) for tile generation, imagery download, and label processing.

### Annotations

Labels are crowd-sourced building footprints from OpenStreetMap, created and validated by humanitarian mappers through HOT Tasking Manager campaigns. Each project goes through:
1. **Mapping phase:** Volunteers digitize building footprints from aerial imagery
2. **Validation phase:** Experienced mappers review and correct the mapped features

Only projects with >= 95% validation are included, ensuring high annotation quality.

## Considerations for Using the Data

### Social Impact

This dataset supports humanitarian applications including disaster response, risk assessment, and development planning. Building footprint data is critical for estimating population exposure, damage assessment, and resource allocation during natural disasters.

### Known Limitations

- **Temporal mismatch:** OSM data reflects current building footprints while OAM imagery may be from different dates. Buildings constructed or destroyed between imagery capture and OSM editing may cause label noise.
- **Geographic bias:** Project locations are concentrated in disaster-affected and developing regions where HOT operates.
- **Label completeness:** While validated, some buildings may be missed or incorrectly mapped in OSM.
- **Imagery quality:** OAM imagery varies in resolution, cloud cover, and viewing angle across projects.

### Licensing

This dataset uses a dual-license model:

- **Imagery (image tiles):** CC-BY 4.0 - sourced from OpenAerialMap. All imagery uploaded to OAM is licensed as CC-BY 4.0, with attribution to "contributors of Open Imagery Network." Original copyright remains with the imagery provider.
- **Labels (building footprints):** ODbL 1.0 - sourced from OpenStreetMap. Requires attribution to "OpenStreetMap contributors" and share-alike for derivative databases.
- **Dataset tooling:** GPL-3.0-or-later

## Additional Information

### Dataset Curators

Built using the [hot-oam-dataset](https://github.com/hotosm/building-tm-ai-datasets) tool by HOT.

### Citation

```bibtex
@misc{hot_building_segmentation_2026,
  title={HOT Building Segmentation Dataset},
  author={Humanitarian OpenStreetMap Team},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/hotosm/building-segmentation}}
}
```

### Contributions

Powered by data from [OpenStreetMap](https://www.openstreetmap.org/) contributors, [OpenAerialMap](https://openaerialmap.org/), and the [HOT Tasking Manager](https://tasks.hotosm.org/) volunteer community.
