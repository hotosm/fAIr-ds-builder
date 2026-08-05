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

import geopandas as gpd
import httpx
from geomltoolkits.downloader import osm as osm_downloader
from geomltoolkits.downloader import tms as tms_downloader
from geomltoolkits.raster.burn import burn_labels

from hot_oam_dataset.feature_spec import DEFAULT_FEATURE_TYPE, FeatureSpec, resolve_feature_spec
from hot_oam_dataset.tm_client import TaskingManagerProject

logger = logging.getLogger(__name__)

TILE_ZOOM = 19
TILE_PREFIX = "OAM"
TILE_SIZE_PIXELS = 256
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
    feature_type: str = DEFAULT_FEATURE_TYPE
    burn_binary_masks: bool = True
    burn_value: int = 255
    label_buffer_meters: float = 0.0
    tm_mapping_type: str | None = None
    geometry_types: list[str] | None = None

    @property
    def feature_spec(self) -> FeatureSpec:
        return resolve_feature_spec(self.feature_type, self.tm_mapping_type, self.geometry_types)


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
            if not response.is_error:
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

    _save_project_metadata(project, project_dir, config)

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

    logger.info("Downloading OSM label data for project #%d", project.project_id)

    spec = config.feature_spec

    logger.info(
        "Requesting OSM tag '%s' as %s geometry",
        spec.osm_tag,
        "/".join(spec.geometry_types),
    )

    await osm_downloader.download_osm_data(
        geojson=str(tiles_geojson_path),
        out=osm_work_dir,
        feature_type=spec.osm_tag,
        geometry_types=list(spec.geometry_types),
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


def _write_buffered_labels(labels_path: Path, buffer_meters: float) -> Path:
    """Widen label geometries in a metric CRS so line features rasterize to a usable width."""
    labels = gpd.read_file(labels_path)

    if labels.empty:
        logger.info("No labels to buffer in %s", labels_path)
        return labels_path

    metric = labels.to_crs(labels.estimate_utm_crs())
    metric["geometry"] = metric.geometry.buffer(buffer_meters)

    buffered_path = labels_path.with_name("osm-result-buffered.geojson")
    metric.to_crs(labels.crs).to_file(buffered_path, driver="GeoJSON")

    logger.info("Buffered labels by %.1fm for mask burning: %s", buffer_meters, buffered_path)
    return buffered_path


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

    burn_source = labels_path
    if config.label_buffer_meters > 0:
        burn_source = _write_buffered_labels(labels_path, config.label_buffer_meters)

    burn_labels(
        labels_path=str(burn_source),
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
    total_label_features = 0

    if not labels_dir.exists():
        return removed_count, valid_pairs, empty_label_pairs, total_label_features

    for label in list(labels_dir.glob(f"{prefix}-*.geojson")):
        if label.stem not in valid_chip_stems:
            label.unlink()
            removed_count += 1
            continue

        data = json.loads(label.read_text())
        feature_count = len(data.get("features", []))
        total_label_features += feature_count

        if feature_count == 0:
            empty_label_pairs += 1

        valid_pairs += 1

    return removed_count, valid_pairs, empty_label_pairs, total_label_features


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

    removed_orphans, valid_pairs, empty_label_pairs, total_label_features = _remove_orphan_labels(
        labels_dir, valid_chip_stems, prefix
    )

    removed_unpaired = _remove_unpaired_chips(chips_dir, labels_dir, valid_chip_stems)

    logger.info(
        "Project #%d: %d valid pairs (%d with labels, %d empty), "
        "%d label features, removed %d orphan labels, %d unpaired chips, "
        "%d corrupt chips",
        project_id,
        valid_pairs,
        valid_pairs - empty_label_pairs,
        empty_label_pairs,
        total_label_features,
        removed_orphans,
        removed_unpaired,
        removed_corrupt,
    )

    size_bytes = _compute_directory_size(project_dir, ["chips", "labels"])

    return {
        "valid_pairs": valid_pairs,
        "pairs_with_features": valid_pairs - empty_label_pairs,
        "pairs_without_features": empty_label_pairs,
        "total_label_features": total_label_features,
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


def _save_project_metadata(
    project: TaskingManagerProject, project_dir: Path, config: DatasetConfig
):
    spec = config.feature_spec
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
        "feature_type": spec.osm_tag,
        "tm_mapping_type": spec.tm_mapping_type,
        "geometry_types": list(spec.geometry_types),
        "tile_prefix": config.prefix,
        "zoom": config.zoom,
        "tile_size_pixels": TILE_SIZE_PIXELS,
        "label_buffer_meters": config.label_buffer_meters,
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def build_all_projects(
    projects: list[TaskingManagerProject],
    config: DatasetConfig,
    dataset_version: str = "1.0.0",
) -> list[int]:
    logger.info("Building dataset for %d projects", len(projects))
    failed_project_ids: list[int] = []

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
            failed_project_ids.append(project.project_id)

    aggregate_dataset_stats(config.output_dir, dataset_version=dataset_version)

    if failed_project_ids:
        logger.error(
            "%d of %d projects failed: %s",
            len(failed_project_ids),
            len(projects),
            failed_project_ids,
        )

    logger.info("Dataset build complete. Output: %s", config.output_dir)
    return failed_project_ids


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


def _distinct_metadata_values(collected: list[tuple[dict, dict]], key: str, default) -> list:
    values = {metadata.get(key, default) for metadata, _ in collected}
    return sorted(values, key=str)


def _single_or_mixed(values: list):
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def _project_report(metadata: dict, stats: dict, area_sq_km: float) -> dict:
    return {
        "project_id": metadata["project_id"],
        "name": metadata["name"],
        "country": metadata.get("country", "Unknown"),
        "organisation": metadata.get("organisation", ""),
        "feature_type": metadata.get("feature_type", DEFAULT_FEATURE_TYPE),
        "valid_pairs": stats["valid_pairs"],
        "pairs_with_features": stats["pairs_with_features"],
        "pairs_without_features": stats["pairs_without_features"],
        "total_label_features": stats["total_label_features"],
        "area_sq_km": area_sq_km,
        "size_mb": round(stats.get("size_bytes", 0) / (1024 * 1024), 1),
    }


def _accumulate_country_details(project_reports: list[dict]) -> dict[str, dict]:
    projects: Counter[str] = Counter()
    tiles: Counter[str] = Counter()
    features: Counter[str] = Counter()
    area: Counter[str] = Counter()

    for report in project_reports:
        country = report["country"]
        projects[country] += 1
        tiles[country] += report["valid_pairs"]
        features[country] += report["total_label_features"]
        area[country] += report["area_sq_km"]

    return {
        country: {
            "projects": projects[country],
            "tiles": tiles[country],
            "features": features[country],
            "area_sq_km": round(area[country], 2),
        }
        for country in projects
    }


def _build_aggregate_report(collected: list[tuple[dict, dict]], dataset_version: str) -> dict:
    project_reports = [
        _project_report(metadata, stats, _bbox_area_sq_km(metadata.get("bbox", [])))
        for metadata, stats in collected
    ]

    total_pairs = sum(r["valid_pairs"] for r in project_reports)
    total_label_features = sum(r["total_label_features"] for r in project_reports)
    total_size_bytes = sum(stats.get("size_bytes", 0) for _, stats in collected)

    countries = Counter(r["country"] for r in project_reports)
    organisations = Counter(metadata.get("organisation", "Unknown") for metadata, _ in collected)
    country_details = _accumulate_country_details(project_reports)

    feature_types = _distinct_metadata_values(collected, "feature_type", DEFAULT_FEATURE_TYPE)
    label_noun = (
        resolve_feature_spec(feature_types[0]).label_noun if len(feature_types) == 1 else "features"
    )

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "dataset_version": dataset_version,
        "total_projects": len(collected),
        "total_tile_pairs": total_pairs,
        "tiles_with_features": sum(r["pairs_with_features"] for r in project_reports),
        "tiles_without_features": sum(r["pairs_without_features"] for r in project_reports),
        "total_label_features": total_label_features,
        "avg_features_per_tile": (
            round(total_label_features / total_pairs, 1) if total_pairs else 0
        ),
        "total_area_sq_km": round(sum(r["area_sq_km"] for r in project_reports), 2),
        "feature_type": _single_or_mixed(feature_types),
        "label_noun": label_noun,
        "tile_size_pixels": _single_or_mixed(
            _distinct_metadata_values(collected, "tile_size_pixels", TILE_SIZE_PIXELS)
        ),
        "zoom_level": _single_or_mixed(_distinct_metadata_values(collected, "zoom", TILE_ZOOM)),
        "tile_prefix": TILE_PREFIX,
        "imagery_source": "OpenAerialMap",
        "label_source": "OpenStreetMap",
        "label_format": "GeoJSON",
        "total_size_gb": round(total_size_bytes / (1024**3), 2),
        "countries": dict(countries.most_common()),
        "country_details": dict(sorted(country_details.items(), key=lambda x: -x[1]["tiles"])),
        "organisations": dict(organisations.most_common()),
        "projects": project_reports,
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
    noun = stats.get("label_noun", "features")
    print(f"  Feature type:          {stats.get('feature_type')}")
    print(f"  Tiles with {noun}:".ljust(25) + f"{stats['tiles_with_features']:,}")
    print(f"  Tiles without:         {stats['tiles_without_features']:,}")
    print(f"  Label {noun}:".ljust(25) + f"{stats['total_label_features']:,}")
    print(f"  Avg {noun}/tile:".ljust(25) + f"{stats['avg_features_per_tile']}")
    tile_px = stats["tile_size_pixels"]
    print(f"  Tile size:             {tile_px}x{tile_px} px")
    print(f"  Zoom level:            {stats['zoom_level']}")
    print(f"  Dataset size:          {stats['total_size_gb']} GB")
    print(f"  Countries:             {len(stats['countries'])}")
    for country, count in list(stats["countries"].items())[:10]:
        print(f"    {country}: {count} projects")
    print(f"{'=' * width}\n")
