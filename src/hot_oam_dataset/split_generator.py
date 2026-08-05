import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLIT_NAMES = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
DEFAULT_RATIOS = {SPLIT_TRAIN: 0.80, SPLIT_VALIDATION: 0.10, SPLIT_TEST: 0.10}
SPLITS_FILENAME = "splits.json"


@dataclass
class ProjectCluster:
    imagery_url: str
    project_ids: list[int]
    total_tiles: int


@dataclass
class ProjectDetail:
    tiles: int
    features: int
    country: str


def _load_project_details(dataset_dir: Path) -> dict[int, ProjectDetail]:
    stats_path = dataset_dir / "dataset_stats.json"
    if not stats_path.exists():
        return {}

    stats = json.loads(stats_path.read_text())
    return {
        p["project_id"]: ProjectDetail(
            tiles=p["valid_pairs"],
            features=p.get("total_label_features", 0),
            country=p.get("country", ""),
        )
        for p in stats.get("projects", [])
    }


def _load_project_tile_counts(dataset_dir: Path) -> dict[int, int]:
    return {pid: d.tiles for pid, d in _load_project_details(dataset_dir).items()}


def _load_project_imagery_urls(dataset_dir: Path) -> dict[int, str]:
    summary_path = dataset_dir / "projects_summary.json"
    if not summary_path.exists():
        return {}

    projects = json.loads(summary_path.read_text())
    return {p["project_id"]: p["imagery_url"] for p in projects}


def build_clusters(dataset_dir: Path) -> list[ProjectCluster]:
    tile_counts = _load_project_tile_counts(dataset_dir)
    imagery_urls = _load_project_imagery_urls(dataset_dir)

    url_to_projects: dict[str, list[int]] = defaultdict(list)
    for project_id, url in imagery_urls.items():
        if project_id in tile_counts:
            url_to_projects[url].append(project_id)

    clusters = []
    for url, project_ids in url_to_projects.items():
        total_tiles = sum(tile_counts.get(pid, 0) for pid in project_ids)
        clusters.append(
            ProjectCluster(
                imagery_url=url,
                project_ids=sorted(project_ids),
                total_tiles=total_tiles,
            )
        )

    clusters.sort(key=lambda c: c.total_tiles, reverse=True)
    return clusters


def assign_splits(
    clusters: list[ProjectCluster],
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    ratios = ratios or DEFAULT_RATIOS
    total_tiles = sum(c.total_tiles for c in clusters)

    if total_tiles == 0:
        return {}

    split_tiles: dict[str, int] = dict.fromkeys(SPLIT_NAMES, 0)
    split_targets = {name: ratio * total_tiles for name, ratio in ratios.items()}
    assignments: dict[int, str] = {}

    for cluster in clusters:
        best_split = _pick_best_fit_split(split_tiles, split_targets, cluster.total_tiles)
        for pid in cluster.project_ids:
            assignments[pid] = best_split
        split_tiles[best_split] += cluster.total_tiles

    return assignments


def _pick_best_fit_split(
    current_tiles: dict[str, int],
    targets: dict[str, float],
    cluster_tiles: int,
) -> str:
    deficits = {s: targets[s] - current_tiles[s] for s in SPLIT_NAMES}

    fits = [s for s in SPLIT_NAMES if deficits[s] >= cluster_tiles]
    if fits:
        return min(fits, key=lambda s: deficits[s] - cluster_tiles)

    return max(SPLIT_NAMES, key=lambda s: deficits[s])


def generate_splits(
    dataset_dir: Path,
    ratios: dict[str, float] | None = None,
) -> dict[int, str]:
    clusters = build_clusters(dataset_dir)

    if not clusters:
        logger.warning("No project clusters found, cannot generate splits")
        return {}

    logger.info(
        "Built %d clusters from %d projects",
        len(clusters),
        sum(len(c.project_ids) for c in clusters),
    )
    for cluster in clusters[:5]:
        logger.debug(
            "  Cluster %s: %d tiles, projects %s",
            cluster.imagery_url[-40:],
            cluster.total_tiles,
            cluster.project_ids,
        )

    assignments = assign_splits(clusters, ratios)
    splits_path = dataset_dir / SPLITS_FILENAME
    splits_path.write_text(json.dumps(assignments, indent=2, sort_keys=True))

    split_summary = _summarize_splits(assignments, dataset_dir)
    _log_split_summary(split_summary)

    logger.info("Splits saved to %s", splits_path)
    return assignments


def load_splits(dataset_dir: Path) -> dict[int, str]:
    splits_path = dataset_dir / SPLITS_FILENAME
    if not splits_path.exists():
        return {}

    raw = json.loads(splits_path.read_text())
    return {int(k): v for k, v in raw.items()}


def compute_split_stats(dataset_dir: Path) -> dict[str, dict] | None:
    assignments = load_splits(dataset_dir)
    if not assignments:
        return None
    return _summarize_splits(assignments, dataset_dir)


def _summarize_splits(
    assignments: dict[int, str],
    dataset_dir: Path,
) -> dict[str, dict]:
    details = _load_project_details(dataset_dir)
    total_tiles = sum(details[pid].tiles for pid in assignments if pid in details)

    summary: dict[str, dict] = {}
    for split_name in SPLIT_NAMES:
        project_ids = [pid for pid, s in assignments.items() if s == split_name]
        tiles = sum(details[pid].tiles for pid in project_ids if pid in details)
        features = sum(details[pid].features for pid in project_ids if pid in details)
        countries = sorted(
            {details[pid].country for pid in project_ids if pid in details and details[pid].country}
        )
        pct = (tiles / total_tiles * 100) if total_tiles else 0
        summary[split_name] = {
            "projects": len(project_ids),
            "tiles": tiles,
            "features": features,
            "countries": countries,
            "percent": round(pct, 1),
        }

    return summary


def _log_split_summary(summary: dict[str, dict]):
    logger.info("Split assignment summary:")
    for split_name in SPLIT_NAMES:
        info = summary[split_name]
        logger.info(
            "  %s: %d projects, %d tiles, %d label features, %d countries (%.1f%%)",
            split_name,
            info["projects"],
            info["tiles"],
            info["features"],
            len(info["countries"]),
            info["percent"],
        )
