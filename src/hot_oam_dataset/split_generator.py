import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
from shapely.geometry import box
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
# Test beats validation beats train; buffer and shared-feature rules drop the losing tile
SPLIT_RANK = {SPLIT_TEST: 0, SPLIT_VALIDATION: 1, SPLIT_TRAIN: 2}
SPLITS_FILENAME = "splits.json"
EXCLUDED_BY_BUFFER = "buffer"
EXCLUDED_BY_SHARED_FEATURE = "shared_feature"

TILE_ID_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<x>\d+)-(?P<y>\d+)-(?P<z>\d+)$")


@dataclass(frozen=True)
class SplitSettings:
    """How tiles are split; read from the `split` key of a dataset config."""

    regions_per_project: int = 10
    validation_regions: int = 1
    test_regions: int = 1
    buffer_meters: float = 0.0
    exclude_shared_features: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict) -> "SplitSettings":
        unknown = set(mapping) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown split settings: {sorted(unknown)}")
        settings = cls(**mapping)
        held_out = settings.validation_regions + settings.test_regions
        if settings.validation_regions < 1 or settings.test_regions < 1:
            raise ValueError("split needs at least one validation and one test region")
        if settings.regions_per_project <= held_out:
            raise ValueError(
                f"regions_per_project ({settings.regions_per_project}) must exceed the "
                f"{held_out} held-out regions, so every project keeps training tiles"
            )
        if settings.buffer_meters < 0:
            raise ValueError("buffer_meters must not be negative")
        if not isinstance(settings.exclude_shared_features, bool):
            raise ValueError("exclude_shared_features must be true or false")
        return settings


@dataclass(frozen=True)
class TileRef:
    tile_id: str
    project_id: int
    x: int
    y: int
    z: int
    label_path: Path


@dataclass
class SplitAssignment:
    """Split of every tile, and the tiles the buffer or shared-feature rule leaves out."""

    settings: SplitSettings
    tiles: dict[str, str]
    excluded: dict[str, str]

    def split_for(self, tile_id: str) -> str | None:
        return self.tiles.get(tile_id)


def collect_tiles(dataset_dir: Path) -> list[TileRef]:
    tiles: list[TileRef] = []

    for project_dir in sorted(dataset_dir.glob("project_*")):
        metadata_path = project_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        project_id = json.loads(metadata_path.read_text())["project_id"]
        for label_path in sorted((project_dir / "labels").glob("*.geojson")):
            match = TILE_ID_PATTERN.match(label_path.stem)
            if match:
                x, y, z = (int(match.group(key)) for key in ("x", "y", "z"))
                tiles.append(TileRef(label_path.stem, project_id, x, y, z, label_path))

    return tiles


def tile_footprints(tiles: list[TileRef]) -> gpd.GeoDataFrame:
    """Tile squares in the local UTM zone, so distances are ground metres."""
    geometries = [box(*mercantile.bounds(tile.x, tile.y, tile.z)) for tile in tiles]
    frame = gpd.GeoDataFrame(
        {"tile_id": [tile.tile_id for tile in tiles]}, geometry=geometries, crs=4326
    )
    return frame.to_crs(frame.estimate_utm_crs())


def _project_regions(tiles: list[TileRef], settings: SplitSettings, seed: int) -> dict[str, str]:
    """Cluster a project's tiles into contiguous regions and hold some out whole."""
    if len(tiles) < settings.regions_per_project:
        logger.warning(
            "Project %d has %d tiles, fewer than %d regions; all go to train",
            seed,
            len(tiles),
            settings.regions_per_project,
        )
        return dict.fromkeys((tile.tile_id for tile in tiles), SPLIT_TRAIN)

    centres = tile_footprints(tiles).geometry.centroid
    coordinates = np.column_stack([centres.x, centres.y])
    regions = KMeans(settings.regions_per_project, n_init=10, random_state=seed).fit_predict(
        coordinates
    )

    # k-means numbers its regions arbitrarily, so the first ones are as good a pick as any
    held_out = dict.fromkeys(range(settings.test_regions), SPLIT_TEST)
    validation_end = settings.test_regions + settings.validation_regions
    held_out |= dict.fromkeys(range(settings.test_regions, validation_end), SPLIT_VALIDATION)
    return {
        tile.tile_id: held_out.get(int(region), SPLIT_TRAIN)
        for tile, region in zip(tiles, regions, strict=True)
    }


def _buffer_exclusions(
    tiles: list[TileRef], splits: dict[str, str], buffer_meters: float
) -> set[str]:
    """Tiles closer than the buffer to a kept tile of a winning split, measured per UTM zone."""
    footprints = gpd.GeoDataFrame(
        {
            "tile_id": [tile.tile_id for tile in tiles],
            "project_id": [tile.project_id for tile in tiles],
            "rank": [SPLIT_RANK[splits[tile.tile_id]] for tile in tiles],
        },
        geometry=[box(*mercantile.bounds(tile.x, tile.y, tile.z)) for tile in tiles],
        crs=4326,
    )
    zones = {
        project_id: project_tiles.estimate_utm_crs().srs
        for project_id, project_tiles in footprints.groupby("project_id")
    }
    footprints["zone"] = footprints["project_id"].map(zones)

    excluded: set[str] = set()
    for zone, zone_tiles in footprints.groupby("zone"):
        metric = zone_tiles.to_crs(zone)
        for rank in sorted(set(metric["rank"]))[:-1]:
            kept = metric[~metric["tile_id"].isin(excluded)]
            stronger = kept[kept["rank"] == rank]
            weaker = kept[kept["rank"] > rank]
            near = weaker.sjoin(stronger, predicate="dwithin", distance=buffer_meters)
            excluded |= set(near["tile_id_left"])
    return excluded


def _tile_features(tile: TileRef) -> set[tuple[str, int]]:
    features = json.loads(tile.label_path.read_text())["features"]
    return {(f["properties"]["osm_type"], f["properties"]["osm_id"]) for f in features}


def _shared_feature_exclusions(tiles: list[TileRef], splits: dict[str, str]) -> set[str]:
    """Tiles holding part of a feature that also lies in a winning split."""
    features = {tile.tile_id: _tile_features(tile) for tile in tiles}
    strongest: dict[tuple[str, int], int] = {}
    for tile in tiles:
        for feature in features[tile.tile_id]:
            rank = SPLIT_RANK[splits[tile.tile_id]]
            strongest[feature] = min(rank, strongest.get(feature, rank))
    return {
        tile.tile_id
        for tile in tiles
        if any(strongest[f] < SPLIT_RANK[splits[tile.tile_id]] for f in features[tile.tile_id])
    }


def assign_splits(tiles: list[TileRef], settings: SplitSettings) -> SplitAssignment:
    by_project: dict[int, list[TileRef]] = defaultdict(list)
    for tile in tiles:
        by_project[tile.project_id].append(tile)

    splits: dict[str, str] = {}
    for project_id in sorted(by_project):
        splits |= _project_regions(by_project[project_id], settings, seed=project_id)

    excluded: dict[str, str] = {}
    if settings.buffer_meters > 0:
        buffered = _buffer_exclusions(tiles, splits, settings.buffer_meters)
        excluded |= dict.fromkeys(buffered, EXCLUDED_BY_BUFFER)
    if settings.exclude_shared_features:
        kept = [tile for tile in tiles if tile.tile_id not in excluded]
        shared = _shared_feature_exclusions(kept, splits)
        excluded |= dict.fromkeys(shared, EXCLUDED_BY_SHARED_FEATURE)

    kept_splits = {tile_id: name for tile_id, name in splits.items() if tile_id not in excluded}
    return SplitAssignment(settings=settings, tiles=kept_splits, excluded=excluded)


def generate_splits(dataset_dir: Path, settings: SplitSettings) -> SplitAssignment | None:
    tiles = collect_tiles(dataset_dir)
    if not tiles:
        logger.warning("No tiles found under %s, cannot generate splits", dataset_dir)
        return None

    assignment = assign_splits(tiles, settings)
    payload = {
        "settings": asdict(settings),
        "tiles": dict(sorted(assignment.tiles.items())),
        "excluded": dict(sorted(assignment.excluded.items())),
    }
    (dataset_dir / SPLITS_FILENAME).write_text(json.dumps(payload, indent=2))

    _log_summary(tiles, assignment)
    logger.info("Splits saved to %s", dataset_dir / SPLITS_FILENAME)
    return assignment


def load_splits(dataset_dir: Path) -> SplitAssignment | None:
    splits_path = dataset_dir / SPLITS_FILENAME
    if not splits_path.exists():
        return None

    payload = json.loads(splits_path.read_text())
    if "settings" not in payload:
        raise ValueError(f"{splits_path} uses an older format; rerun --generate-splits")
    return SplitAssignment(
        settings=SplitSettings.from_mapping(payload["settings"]),
        tiles=payload["tiles"],
        excluded=payload["excluded"],
    )


def _log_summary(tiles: list[TileRef], assignment: SplitAssignment) -> None:
    per_split = Counter(assignment.tiles.values())
    total = len(tiles)
    logger.info("Split assignment over %d tiles:", total)
    for name in SPLIT_NAMES:
        logger.info("  %s: %d tiles (%.1f%%)", name, per_split[name], 100 * per_split[name] / total)
    for reason, count in Counter(assignment.excluded.values()).items():
        logger.info("  left out by %s rule: %d tiles", reason, count)
