import asyncio
import json
import logging
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from geomltoolkits.downloader import osm as osm_downloader
from geomltoolkits.downloader import tms as tms_downloader
from geomltoolkits.raster.burn import burn_labels

from hot_oam_dataset.tm_client import TaskingManagerProject

logger = logging.getLogger(__name__)

TILE_ZOOM = 19
TILE_PREFIX = "OAM"
MINIMUM_VALID_CHIP_BYTES = 5000
PROBE_MAX_RETRIES = 3
PROBE_TIMEOUT = 15
PROBE_BACKOFF_BASE = 2


@dataclass
class DatasetConfig:
    output_dir: Path
    zoom: int = TILE_ZOOM
    prefix: str = TILE_PREFIX
    georeference: bool = True
    feature_type: str = "building"
    burn_binary_masks: bool = True
    burn_value: int = 255


def _lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def probe_imagery_url(
    imagery_url: str,
    bbox: list[float],
    zoom: int,
    max_retries: int = PROBE_MAX_RETRIES,
) -> bool:
    if not bbox or len(bbox) < 4:
        return True

    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    x, y = _lon_lat_to_tile(lon, lat, zoom)

    probe_url = imagery_url.replace("{z}", str(zoom)).replace("{x}", str(x)).replace("{y}", str(y))

    for attempt in range(1, max_retries + 1):
        try:
            response = httpx.get(probe_url, timeout=PROBE_TIMEOUT, follow_redirects=True)
            if response.status_code < 400:
                return True
            logger.debug(
                "Probe attempt %d/%d got HTTP %d for %s",
                attempt,
                max_retries,
                response.status_code,
                probe_url,
            )
        except httpx.RequestError as exc:
            logger.debug(
                "Probe attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                probe_url,
                exc,
            )

        if attempt < max_retries:
            time.sleep(PROBE_BACKOFF_BASE**attempt)

    return False


def build_project_dataset(project: TaskingManagerProject, config: DatasetConfig):
    project_dir = config.output_dir / f"project_{project.project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)

    if not probe_imagery_url(project.imagery_url, project.bbox, config.zoom):
        logger.warning(
            "Project #%d imagery URL appears dead, skipping: %s",
            project.project_id,
            project.imagery_url,
        )
        return

    _save_project_metadata(project, project_dir)

    aoi_path = project_dir / "aoi.geojson"
    aoi_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"project_id": project.project_id},
                "geometry": project.area_of_interest,
            }
        ],
    }
    aoi_path.write_text(json.dumps(aoi_geojson))

    asyncio.run(_download_chips_and_labels(project, project_dir, aoi_path, config))


async def _download_chips_and_labels(
    project: TaskingManagerProject,
    project_dir: Path,
    aoi_path: Path,
    config: DatasetConfig,
):
    logger.info(
        "Downloading OAM tiles for project #%d at zoom %d",
        project.project_id,
        config.zoom,
    )

    await tms_downloader.download_tiles(
        tms=project.imagery_url,
        zoom=config.zoom,
        out=str(project_dir),
        geojson=str(aoi_path),
        within=True,
        georeference=config.georeference,
        dump_tile_geometries_as_geojson=True,
        prefix=config.prefix,
    )

    tiles_geojson_path = project_dir / "tiles.geojson"
    if not tiles_geojson_path.exists():
        logger.error("tiles.geojson not found for project #%d", project.project_id)
        return

    osm_work_dir = str(project_dir / "_osm_tmp")

    logger.info("Downloading OSM building data for project #%d", project.project_id)

    await osm_downloader.download_osm_data(
        geojson=str(tiles_geojson_path),
        out=osm_work_dir,
        feature_type=config.feature_type,
        dump_results=True,
        split_output_by_tiles=True,
        split_prefix=config.prefix,
    )

    _flatten_label_output(project_dir)

    if config.burn_binary_masks:
        _burn_project_binary_masks(project_dir, config)

    stats = validate_and_clean_project(project_dir, config.prefix, project.project_id)
    _update_metadata_with_stats(project_dir, stats)

    logger.info("Completed project #%d", project.project_id)


def _flatten_label_output(project_dir: Path):
    osm_work_dir = project_dir / "_osm_tmp"
    labels_dir = project_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    split_geojson_dir = osm_work_dir / "split" / "geojson"
    if split_geojson_dir.exists():
        for geojson_file in split_geojson_dir.glob("*.geojson"):
            geojson_file.rename(labels_dir / geojson_file.name)

    osm_result = osm_work_dir / "osm-result.geojson"
    if osm_result.exists():
        osm_result.rename(labels_dir / "osm-result.geojson")

    shutil.rmtree(osm_work_dir, ignore_errors=True)


def _burn_project_binary_masks(project_dir: Path, config: DatasetConfig):
    labels_path = project_dir / "labels" / "osm-result.geojson"
    chips_dir = project_dir / "chips"
    masks_dir = project_dir / "masks"

    if not labels_path.exists() or not chips_dir.exists():
        logger.warning(
            "Skipping mask burn for %s because required inputs are missing",
            project_dir.name,
        )
        return

    burn_labels(
        labels_path=str(labels_path),
        chips_dir=str(chips_dir),
        output_dir=str(masks_dir),
        burn_value=config.burn_value,
    )

    logger.info("Generated binary masks in %s", masks_dir)


def _remove_corrupt_chips(chips_dir: Path, prefix: str) -> tuple[set[str], int]:
    valid_stems: set[str] = set()
    removed_count = 0

    for chip in list(chips_dir.glob(f"{prefix}-*.tif")):
        if chip.stat().st_size < MINIMUM_VALID_CHIP_BYTES:
            chip.unlink()
            aux = chip.with_suffix(".tif.aux.xml")
            if aux.exists():
                aux.unlink()
            removed_count += 1
        else:
            valid_stems.add(chip.stem)

    return valid_stems, removed_count


def _remove_orphan_labels(
    labels_dir: Path, valid_chip_stems: set[str], prefix: str
) -> tuple[int, int, int, int]:
    removed_count = 0
    valid_pairs = 0
    empty_label_pairs = 0
    total_building_polygons = 0

    if not labels_dir.exists():
        return removed_count, valid_pairs, empty_label_pairs, total_building_polygons

    for label in list(labels_dir.glob(f"{prefix}-*.geojson")):
        if label.stem not in valid_chip_stems:
            label.unlink()
            removed_count += 1
            continue

        data = json.loads(label.read_text())
        feature_count = len(data.get("features", []))
        total_building_polygons += feature_count

        if feature_count == 0:
            empty_label_pairs += 1

        valid_pairs += 1

    return removed_count, valid_pairs, empty_label_pairs, total_building_polygons


def _remove_unpaired_chips(chips_dir: Path, labels_dir: Path, valid_chip_stems: set[str]) -> int:
    removed_count = 0

    for stem in list(valid_chip_stems):
        label_path = labels_dir / f"{stem}.geojson" if labels_dir.exists() else None
        if label_path and not label_path.exists():
            chip_path = chips_dir / f"{stem}.tif"
            if chip_path.exists():
                chip_path.unlink()
                aux = chip_path.with_suffix(".tif.aux.xml")
                if aux.exists():
                    aux.unlink()
                removed_count += 1

    return removed_count


def validate_and_clean_project(project_dir: Path, prefix: str, project_id: int) -> dict:
    chips_dir = project_dir / "chips"
    labels_dir = project_dir / "labels"

    valid_chip_stems, removed_corrupt = _remove_corrupt_chips(chips_dir, prefix)

    if removed_corrupt:
        logger.info("Project #%d: removed %d corrupt/empty chips", project_id, removed_corrupt)

    removed_orphans, valid_pairs, empty_label_pairs, total_building_polygons = (
        _remove_orphan_labels(labels_dir, valid_chip_stems, prefix)
    )

    removed_unpaired = _remove_unpaired_chips(chips_dir, labels_dir, valid_chip_stems)

    logger.info(
        "Project #%d: %d valid pairs (%d with buildings, %d empty), "
        "%d building polygons, removed %d orphan labels, %d unpaired chips, "
        "%d corrupt chips",
        project_id,
        valid_pairs,
        valid_pairs - empty_label_pairs,
        empty_label_pairs,
        total_building_polygons,
        removed_orphans,
        removed_unpaired,
        removed_corrupt,
    )

    size_bytes = _compute_directory_size(project_dir, ["chips", "labels"])

    return {
        "valid_pairs": valid_pairs,
        "pairs_with_buildings": valid_pairs - empty_label_pairs,
        "pairs_without_buildings": empty_label_pairs,
        "total_building_polygons": total_building_polygons,
        "size_bytes": size_bytes,
        "removed_orphan_labels": removed_orphans,
        "removed_unpaired_chips": removed_unpaired,
        "removed_corrupt_chips": removed_corrupt,
    }


def _compute_directory_size(project_dir: Path, subdirs: list[str]) -> int:
    total = 0
    for subdir in subdirs:
        subdirectory = project_dir / subdir
        if subdirectory.exists():
            for file_path in subdirectory.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size
    return total


def _update_metadata_with_stats(project_dir: Path, stats: dict):
    metadata_path = project_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_stats"] = stats
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _save_project_metadata(project: TaskingManagerProject, project_dir: Path):
    metadata = {
        "project_id": project.project_id,
        "name": project.name,
        "status": project.status,
        "percent_validated": project.percent_validated,
        "imagery_url": project.imagery_url,
        "organisation": project.organisation,
        "country": project.country,
        "created": project.created,
        "mapping_types": project.mapping_types,
        "bbox": project.bbox,
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def build_all_projects(
    projects: list[TaskingManagerProject],
    config: DatasetConfig,
    dataset_version: str = "1.0.0",
):
    logger.info("Building dataset for %d projects", len(projects))

    for i, project in enumerate(projects, 1):
        logger.info(
            "[%d/%d] Processing project #%d: %s",
            i,
            len(projects),
            project.project_id,
            project.name,
        )
        try:
            build_project_dataset(project, config)
        except Exception:
            logger.exception("Failed to process project #%d", project.project_id)

    aggregate_dataset_stats(config.output_dir, dataset_version=dataset_version)
    logger.info("Dataset build complete. Output: %s", config.output_dir)


def _collect_per_project_stats(output_dir: Path) -> list[tuple[dict, dict]]:
    collected = []
    for project_dir in sorted(output_dir.glob("project_*")):
        metadata_path = project_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        stats = metadata.get("dataset_stats")
        if not stats:
            continue

        collected.append((metadata, stats))

    return collected


def _bbox_area_sq_km(bbox: list[float]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    west, south, east, north = bbox
    lat_mid = math.radians((south + north) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_mid)
    width_km = abs(east - west) * km_per_deg_lon
    height_km = abs(north - south) * km_per_deg_lat
    return round(width_km * height_km, 2)


def _build_aggregate_report(collected: list[tuple[dict, dict]], dataset_version: str) -> dict:
    total_pairs = 0
    total_with_buildings = 0
    total_without_buildings = 0
    total_building_polygons = 0
    total_size_bytes = 0
    total_area_sq_km = 0.0
    countries: Counter[str] = Counter()
    organisations: Counter[str] = Counter()
    country_tiles: Counter[str] = Counter()
    country_polygons: Counter[str] = Counter()
    country_area: Counter[str] = Counter()
    project_stats = []

    for metadata, stats in collected:
        total_pairs += stats["valid_pairs"]
        total_with_buildings += stats["pairs_with_buildings"]
        total_without_buildings += stats["pairs_without_buildings"]
        total_building_polygons += stats["total_building_polygons"]
        total_size_bytes += stats.get("size_bytes", 0)

        country = metadata.get("country", "Unknown")
        countries[country] += 1
        organisations[metadata.get("organisation", "Unknown")] += 1
        country_tiles[country] += stats["valid_pairs"]
        country_polygons[country] += stats["total_building_polygons"]

        area = _bbox_area_sq_km(metadata.get("bbox", []))
        total_area_sq_km += area
        country_area[country] += area

        project_stats.append(
            {
                "project_id": metadata["project_id"],
                "name": metadata["name"],
                "country": country,
                "organisation": metadata.get("organisation", ""),
                "valid_pairs": stats["valid_pairs"],
                "pairs_with_buildings": stats["pairs_with_buildings"],
                "pairs_without_buildings": stats["pairs_without_buildings"],
                "total_building_polygons": stats["total_building_polygons"],
                "area_sq_km": area,
                "size_mb": round(stats.get("size_bytes", 0) / (1024 * 1024), 1),
            }
        )

    country_details = {
        c: {
            "projects": countries[c],
            "tiles": country_tiles[c],
            "polygons": country_polygons[c],
            "area_sq_km": round(country_area[c], 2),
        }
        for c in countries
    }

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset_version": dataset_version,
        "total_projects": len(collected),
        "total_tile_pairs": total_pairs,
        "tiles_with_buildings": total_with_buildings,
        "tiles_without_buildings": total_without_buildings,
        "total_building_polygons": total_building_polygons,
        "avg_buildings_per_tile": (
            round(total_building_polygons / total_pairs, 1) if total_pairs else 0
        ),
        "total_area_sq_km": round(total_area_sq_km, 2),
        "tile_size_pixels": 256,
        "zoom_level": TILE_ZOOM,
        "tile_prefix": TILE_PREFIX,
        "imagery_source": "OpenAerialMap",
        "label_source": "OpenStreetMap",
        "label_format": "GeoJSON",
        "total_size_gb": round(total_size_bytes / (1024**3), 2),
        "countries": dict(countries.most_common()),
        "country_details": dict(sorted(country_details.items(), key=lambda x: -x[1]["tiles"])),
        "organisations": dict(organisations.most_common()),
        "projects": project_stats,
    }


def aggregate_dataset_stats(output_dir: Path, dataset_version: str = "1.0.0"):
    collected = _collect_per_project_stats(output_dir)
    if not collected:
        return

    aggregate = _build_aggregate_report(collected, dataset_version)

    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("Aggregate dataset stats saved to %s", stats_path)

    _print_dataset_summary(aggregate)


def _print_dataset_summary(stats: dict):
    width = 70
    print(f"\n{'=' * width}")
    print("DATASET SUMMARY")
    print(f"{'=' * width}")
    print(f"  Projects:              {stats['total_projects']}")
    print(f"  Total tile pairs:      {stats['total_tile_pairs']:,}")
    print(f"  Tiles with buildings:  {stats['tiles_with_buildings']:,}")
    print(f"  Tiles without:         {stats['tiles_without_buildings']:,}")
    print(f"  Building polygons:     {stats['total_building_polygons']:,}")
    print(f"  Avg buildings/tile:    {stats['avg_buildings_per_tile']}")
    tile_px = stats["tile_size_pixels"]
    print(f"  Tile size:             {tile_px}x{tile_px} px")
    print(f"  Zoom level:            {stats['zoom_level']}")
    print(f"  Dataset size:          {stats['total_size_gb']} GB")
    print(f"  Countries:             {len(stats['countries'])}")
    for country, count in list(stats["countries"].items())[:10]:
        print(f"    {country}: {count} projects")
    print(f"{'=' * width}\n")
