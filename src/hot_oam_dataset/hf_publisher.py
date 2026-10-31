import json
import logging
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import DatasetCard, HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from hot_oam_dataset.dataset_card import (
    CARD_AUTO_STATS_END,
    CARD_AUTO_STATS_START,
    CardText,
    generic_card_template,
    published_stats,
    read_card_template,
    render_card_text,
    source_project_lines,
)
from hot_oam_dataset.feature_spec import (
    DEFAULT_FEATURE_TYPE,
    dataset_wiki_pages,
    label_class_counts,
)
from hot_oam_dataset.sample_tiles import SAMPLE_TILES_FILENAME, write_sample_tiles
from hot_oam_dataset.split_generator import (
    EXCLUDED_BY_BUFFER,
    EXCLUDED_BY_SHARED_FEATURE,
    SPLIT_NAMES,
    SPLIT_TRAIN,
    SplitAssignment,
    load_splits,
)
from hot_oam_dataset.stac_export import STAC_DIRNAME, write_published_stac

logger = logging.getLogger(__name__)

TILE_ID_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<x>\d+)-(?P<y>\d+)-(?P<z>\d+)$")
TILE_TITLE_PATTERN = re.compile(r"Tile\(x=(\d+), y=(\d+), z=(\d+)\)")
CARD_VERSION_PATTERN = re.compile(r"^\| Version \| ([^,|\s]+)", re.MULTILINE)
CARD_FILENAME = "dataset_card.md"
MAX_CARD_CLASS_ROWS = 15
SAMPLE_SEED = 0
DEFAULT_TILE_PREFIX = "OAM"
HUB_DATASETS_URL = "https://huggingface.co/datasets"


def _project_columns(metadata: dict) -> dict:
    return {
        "project_id": metadata["project_id"],
        "project_url": metadata.get("project_url", ""),
        "project_name": metadata["name"],
        "country": metadata.get("country", ""),
        "organisation": metadata.get("organisation", ""),
        "imagery_url": metadata.get("imagery_url", ""),
        "feature_type": metadata.get("feature_type", DEFAULT_FEATURE_TYPE),
    }


def _tile_record(
    chip_path: Path, project_dir: Path, tile_bounds: dict[str, dict], feature_type: str
) -> dict | None:
    tile_id = chip_path.stem
    label_path = project_dir / "labels" / f"{tile_id}.geojson"
    mask_path = project_dir / "masks" / f"{tile_id}.tif"

    tile_match = TILE_ID_PATTERN.match(tile_id)
    if not label_path.exists() or not tile_match:
        return None

    label_data = json.loads(label_path.read_text())
    bounds = tile_bounds.get(tile_id, {})

    return {
        "image": str(chip_path),
        "mask": str(mask_path) if mask_path.exists() else None,
        "tile_id": tile_id,
        "tile_x": int(tile_match.group("x")),
        "tile_y": int(tile_match.group("y")),
        "tile_z": int(tile_match.group("z")),
        "num_features": len(label_data.get("features", [])),
        "class_counts": json.dumps(dict(label_class_counts(label_data, feature_type))),
        "label_geojson": json.dumps(label_data),
        "bbox_west": bounds.get("west", 0.0),
        "bbox_south": bounds.get("south", 0.0),
        "bbox_east": bounds.get("east", 0.0),
        "bbox_north": bounds.get("north", 0.0),
    }


def _project_parquet_records(project_dir: Path) -> list[dict]:
    metadata_path = project_dir / "metadata.json"
    if not metadata_path.exists():
        return []

    metadata = json.loads(metadata_path.read_text())
    tile_prefix = metadata.get("tile_prefix", DEFAULT_TILE_PREFIX)
    feature_type = metadata.get("feature_type", DEFAULT_FEATURE_TYPE)
    project_columns = _project_columns(metadata)
    tile_bounds = _load_tile_bounds(project_dir / "tiles.geojson", tile_prefix)

    records = []
    for chip_path in sorted((project_dir / "chips").glob(f"{tile_prefix}-*.tif")):
        tile_record = _tile_record(chip_path, project_dir, tile_bounds, feature_type)
        if tile_record:
            records.append({**tile_record, **project_columns})

    return records


def _cap_negatives(records: list[dict], ratio: float) -> tuple[list[dict], dict]:
    """Keep every labelled tile plus at most ratio empty tiles per labelled one, per project."""
    by_project: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_project[record["project_id"]].append(record)

    kept: list[dict] = []
    negatives_seen = 0

    for project_id in sorted(by_project):
        project_records = by_project[project_id]
        positives = [r for r in project_records if r["num_features"] > 0]
        negatives = sorted(
            (r for r in project_records if r["num_features"] == 0), key=lambda r: r["tile_id"]
        )
        negatives_seen += len(negatives)

        allowance = min(len(negatives), int(len(positives) * ratio))
        if allowance and len(negatives) > allowance:
            step = len(negatives) / allowance
            negatives = [negatives[int(i * step)] for i in range(allowance)]
        elif not allowance:
            negatives = []

        kept.extend(positives + negatives)

    kept.sort(key=lambda r: (r["project_id"], r["tile_id"]))
    summary = {
        "ratio": ratio,
        "negatives_before": negatives_seen,
        "negatives_kept": sum(1 for r in kept if r["num_features"] == 0),
        "total_before": len(records),
        "total_kept": len(kept),
    }
    return kept, summary


def _collect_parquet_records(dataset_dir: Path) -> list[dict]:
    records = []
    for project_dir in sorted(dataset_dir.glob("project_*")):
        records.extend(_project_parquet_records(project_dir))
    return records


def _parquet_features() -> Features:
    return Features(
        {
            "image": Image(),
            "mask": Image(),
            "tile_id": Value("string"),
            "tile_x": Value("int32"),
            "tile_y": Value("int32"),
            "tile_z": Value("int32"),
            "project_id": Value("int32"),
            "project_url": Value("string"),
            "project_name": Value("string"),
            "country": Value("string"),
            "organisation": Value("string"),
            "imagery_url": Value("string"),
            "feature_type": Value("string"),
            "num_features": Value("int32"),
            "class_counts": Value("string"),
            "label_geojson": Value("large_string"),
            "bbox_west": Value("float64"),
            "bbox_south": Value("float64"),
            "bbox_east": Value("float64"),
            "bbox_north": Value("float64"),
        }
    )


def _records_to_dataset(records: list[dict], dataset_version: str) -> Dataset:
    features = _parquet_features()
    column_data = {name: [record[name] for record in records] for name in features}
    dataset = Dataset.from_dict(column_data, features=features)
    dataset.info.version = dataset_version
    return dataset


def generate_parquet(
    dataset_dir: Path,
    output_dir: Path | None = None,
    dataset_version: str = "1.0.0",
    negative_ratio: float | None = None,
    card_text: CardText | None = None,
):
    output_dir = output_dir or dataset_dir / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_parquet_records(dataset_dir)

    if not records:
        logger.warning("No valid tile pairs found for Parquet generation")
        return

    sampling = None
    if negative_ratio is not None:
        records, sampling = _cap_negatives(records, negative_ratio)
        logger.info(
            "Capped empty tiles at %.1f:1 per project: %d of %d kept, %d tiles total",
            negative_ratio,
            sampling["negatives_kept"],
            sampling["negatives_before"],
            sampling["total_kept"],
        )

    splits = load_splits(dataset_dir)
    split_stats = None

    if splits:
        split_counts, split_stats, records = _generate_split_parquets(
            records, splits, output_dir, dataset_version
        )
    else:
        logger.info("No splits.json found, generating single Parquet")
        dataset = _records_to_dataset(records, dataset_version)
        dataset.to_parquet(str(output_dir / "data.parquet"))
        split_counts = {SPLIT_TRAIN: len(records)}

    _update_dataset_card(
        dataset_dir,
        records,
        split_counts=split_counts,
        dataset_version=dataset_version,
        sampling=sampling,
        split_stats=split_stats,
        splits=splits,
        card_text=card_text,
    )

    logger.info("Parquet saved to %s (version %s)", output_dir, dataset_version)


def _split_records(records: list[dict], splits: SplitAssignment) -> tuple[dict, int]:
    split_records: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    unassigned = 0

    for record in records:
        split_name = splits.split_for(record["tile_id"])
        if split_name in split_records:
            split_records[split_name].append(record)
        elif record["tile_id"] not in splits.excluded:
            unassigned += 1

    return split_records, unassigned


def split_stats_from_records(split_records: dict[str, list[dict]]) -> dict[str, dict]:
    total = sum(len(rows) for rows in split_records.values())
    return {
        name: {
            "projects": len({r["project_id"] for r in rows}),
            "tiles": len(rows),
            "features": sum(r["num_features"] for r in rows),
            "countries": sorted({r["country"] for r in rows if r["country"]}),
            "percent": round(100 * len(rows) / total, 1) if total else 0,
        }
        for name, rows in split_records.items()
    }


def _generate_split_parquets(
    records: list[dict],
    splits: SplitAssignment,
    output_dir: Path,
    dataset_version: str,
) -> tuple[dict[str, int], dict[str, dict], list[dict]]:
    """Write one Parquet per split; returns the counts, stats and the rows written."""
    split_records, unassigned = _split_records(records, splits)

    if unassigned:
        logger.warning("%d tiles had no split assignment, excluded from output", unassigned)

    split_counts = {}
    for split_name in SPLIT_NAMES:
        records_for_split = split_records[split_name]
        if not records_for_split:
            logger.warning("Split '%s' has no records", split_name)
            continue

        dataset = _records_to_dataset(records_for_split, dataset_version)
        parquet_path = output_dir / f"{split_name}.parquet"
        dataset.to_parquet(str(parquet_path))
        split_counts[split_name] = len(records_for_split)
        logger.info("  %s: %d records -> %s", split_name, len(records_for_split), parquet_path)

    written = [record for name in SPLIT_NAMES for record in split_records[name]]
    return split_counts, split_stats_from_records(split_records), written


def _load_parquet_as_dataset_or_dict(
    parquet_dir: Path,
) -> Dataset | DatasetDict:
    split_parquets = {name: parquet_dir / f"{name}.parquet" for name in SPLIT_NAMES}
    available_splits = {name: path for name, path in split_parquets.items() if path.exists()}

    if available_splits:
        logger.info("Loading split Parquets: %s", list(available_splits.keys()))
        return DatasetDict(
            {name: Dataset.from_parquet(str(path)) for name, path in available_splits.items()}
        )

    single_path = parquet_dir / "data.parquet"
    if single_path.exists():
        return Dataset.from_parquet(str(single_path))

    msg = f"No parquet files found in {parquet_dir}"
    raise FileNotFoundError(msg)


def push_to_hub(
    dataset_dir: Path,
    repo_id: str,
    parquet_dir: Path | None = None,
    dataset_version: str = "1.0.0",
    negative_ratio: float | None = None,
    card_text: CardText | None = None,
):
    parquet_dir = parquet_dir or dataset_dir / "parquet"

    has_any_parquet = any(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else False
    if not has_any_parquet:
        logger.info("Parquet not found, generating first")
        generate_parquet(
            dataset_dir,
            parquet_dir,
            dataset_version=dataset_version,
            negative_ratio=negative_ratio,
            card_text=card_text,
        )

    dataset = _load_parquet_as_dataset_or_dict(parquet_dir)
    _require_card_version(dataset_dir, dataset_version)
    _require_new_release(repo_id, dataset_version)

    logger.info("Pushing to HuggingFace Hub: %s", repo_id)
    _push_dataset_with_card_retry(dataset=dataset, repo_id=repo_id, dataset_version=dataset_version)

    card_path = resolve_dataset_card(dataset_dir)
    logger.info("Uploading dataset card %s", card_path)
    card = DatasetCard(card_path.read_text())
    card.push_to_hub(repo_id, commit_message=f"Update dataset card v{dataset_version}")

    samples_path = dataset_dir / SAMPLE_TILES_FILENAME
    if samples_path.exists():
        HfApi().upload_file(
            path_or_fileobj=samples_path,
            path_in_repo=SAMPLE_TILES_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload sample tiles v{dataset_version}",
        )

    stac_dir = dataset_dir / STAC_DIRNAME
    if stac_dir.is_dir():
        _upload_published_stac(stac_dir, repo_id, dataset_version)

    _tag_hf_revision(repo_id, dataset_version)
    logger.info("Dataset published to %s/%s", HUB_DATASETS_URL, repo_id)


def release_stac_url(repo_id: str, dataset_version: str, route: str) -> str:
    return f"{HUB_DATASETS_URL}/{repo_id}/{route}/v{dataset_version}/{STAC_DIRNAME}"


def _upload_published_stac(stac_dir: Path, repo_id: str, dataset_version: str) -> None:
    """Upload the catalogue with links absolute to the release tag the push creates."""
    # raw/ answers JSON without the relative redirect QGIS cannot follow; LFS assets need resolve/
    links_base_url = release_stac_url(repo_id, dataset_version, "raw")
    assets_base_url = release_stac_url(repo_id, dataset_version, "resolve")
    # Staged beside the dataset so the large binaries are hard-linked, never copied
    with tempfile.TemporaryDirectory(dir=stac_dir.parent) as staging:
        staging_dir = Path(staging)
        write_published_stac(stac_dir, links_base_url, assets_base_url, staging_dir)
        for path in stac_dir.rglob("*"):
            if path.is_file() and path.suffix != ".json":
                target = staging_dir / path.relative_to(stac_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
        logger.info("Uploading STAC catalogue %s with links under %s", stac_dir, links_base_url)
        HfApi().upload_folder(
            folder_path=str(staging_dir),
            path_in_repo=STAC_DIRNAME,
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns=["*"],
            commit_message=f"Upload STAC catalogue v{dataset_version}",
        )


def compress_projects(dataset_dir: Path) -> list[Path]:
    """Create a data.tar.gz per project containing chips/, masks/, and labels/."""
    archive_dirs = ["chips", "masks", "labels"]
    created = []

    for project_dir in sorted(dataset_dir.glob("project_*")):
        if not project_dir.is_dir():
            continue

        archive_path = project_dir / "data.tar.gz"
        dirs_to_archive = [d for d in archive_dirs if (project_dir / d).is_dir()]

        if not dirs_to_archive:
            logger.debug("Skipping %s (no chip/mask/label dirs)", project_dir.name)
            continue

        if archive_path.exists():
            logger.debug("Archive already exists: %s", archive_path)
            created.append(archive_path)
            continue

        logger.info("Compressing %s (%s)", project_dir.name, ", ".join(dirs_to_archive))
        with tarfile.open(archive_path, "w:gz") as tar:
            for dirname in dirs_to_archive:
                tar.add(project_dir / dirname, arcname=dirname)

        created.append(archive_path)

    logger.info("Compressed %d projects", len(created))
    return created


def push_raw_to_hub(
    dataset_dir: Path,
    repo_id: str,
    dataset_version: str = "1.0.0",
    compress: bool = False,
):
    if compress:
        compress_projects(dataset_dir)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    root_level_files = ["projects_map.geojson", "dataset_stats.json", "projects_summary.json"]
    ignore = ["parquet/*", f"{STAC_DIRNAME}/*", ".cache/*", CARD_FILENAME, *root_level_files]
    if compress:
        ignore.extend(["*/chips/*", "*/masks/*", "*/labels/*"])

    logger.info("Uploading raw project data to %s (raw/ on main)", repo_id)
    api.upload_folder(
        folder_path=str(dataset_dir),
        path_in_repo="raw",
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=ignore,
        commit_message=f"Upload raw dataset v{dataset_version}",
    )

    for filename in root_level_files:
        filepath = dataset_dir / filename
        if filepath.exists():
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {filename} v{dataset_version}",
            )

    logger.info("Raw dataset uploaded to %s/%s/tree/main/raw", HUB_DATASETS_URL, repo_id)


def _require_card_version(dataset_dir: Path, dataset_version: str) -> None:
    """The card's stats are written with the Parquet, so they must match the version pushed."""
    card_path = resolve_dataset_card(dataset_dir)
    match = CARD_VERSION_PATTERN.search(card_path.read_text())
    if not match:
        msg = f"{card_path} has no generated stats; run --generate-parquet first"
        raise ValueError(msg)
    if match.group(1) != dataset_version:
        msg = (
            f"{card_path} describes version {match.group(1)}, not {dataset_version}; "
            "rerun --generate-parquet with this --dataset-version"
        )
        raise ValueError(msg)


def release_tags(repo_id: str) -> set[str]:
    # A repo that does not exist yet has no releases; the push creates it
    try:
        refs = HfApi().list_repo_refs(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return set()
    return {ref.name for ref in refs.tags}


def _require_new_release(repo_id: str, dataset_version: str) -> None:
    """Published links point at the release tag, so a pushed version is never pushed again."""
    if f"v{dataset_version}" in release_tags(repo_id):
        msg = f"{repo_id} already has release v{dataset_version}; bump --dataset-version"
        raise ValueError(msg)


def _tag_hf_revision(repo_id: str, dataset_version: str) -> None:
    tag = f"v{dataset_version}"
    HfApi().create_tag(repo_id=repo_id, repo_type="dataset", tag=tag, tag_message=tag)
    logger.info("Tagged HF dataset revision: %s", tag)


def _push_dataset_with_card_retry(
    dataset: Dataset | DatasetDict, repo_id: str, dataset_version: str
) -> None:
    commit_message = f"v{dataset_version}"

    try:
        dataset.push_to_hub(
            repo_id,
            max_shard_size="500MB",
            commit_message=commit_message,
        )
    except TypeError as exc:
        known_metadata_bugs = [
            "unsupported operand type(s) for -=",
            "argument of type 'NoneType' is not iterable",
        ]
        if not any(pattern in str(exc) for pattern in known_metadata_bugs):
            raise

        logger.warning(
            "Detected dataset card metadata bug during push (%s). "
            "Resetting remote README and retrying push.",
            exc,
        )
    else:
        return

    api = HfApi()
    try:
        api.delete_file(
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Reset README to recover from metadata merge bug",
        )
    except HfHubHTTPError as delete_exc:
        logger.warning("No remote README to reset, retrying push anyway: %s", delete_exc)

    dataset.push_to_hub(
        repo_id,
        max_shard_size="500MB",
        commit_message=commit_message,
    )


def _load_tile_bounds(tiles_geojson_path: Path, tile_prefix: str) -> dict[str, dict]:
    if not tiles_geojson_path.exists():
        return {}

    data = json.loads(tiles_geojson_path.read_text())
    bounds = {}

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        title = props.get("title", "")
        match = TILE_TITLE_PATTERN.search(title)
        if not match:
            continue
        tile_id = f"{tile_prefix}-{match.group(1)}-{match.group(2)}-{match.group(3)}"

        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [[]])

        if coords and coords[0]:
            ring = coords[0]
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            bounds[tile_id] = {
                "west": min(lons),
                "south": min(lats),
                "east": max(lons),
                "north": max(lats),
            }

    return bounds


def resolve_dataset_card(dataset_dir: Path) -> Path:
    """Return the dataset's own card, seeding it from the packaged template on first use."""
    card_path = dataset_dir / CARD_FILENAME
    if not card_path.exists():
        card_path.write_text(generic_card_template())
        logger.info("Seeded dataset card %s", card_path)
    return card_path


def _sync_card_dataset_info(card_text: str, split_counts: dict[str, int]) -> str:
    """Regenerate front-matter schema and split sizes; a schema mismatch makes load_dataset fail."""
    card = DatasetCard(card_text)
    dataset_info = card.data.get("dataset_info")
    if not dataset_info:
        return card_text

    # datasets has no public Features-to-YAML helper; its card writer calls this internally
    dataset_info["features"] = _parquet_features()._to_yaml_list()  # noqa: SLF001
    dataset_info["splits"] = [
        {"name": name, "num_examples": count} for name, count in split_counts.items()
    ]
    card.data["dataset_info"] = dataset_info
    # push_to_hub writes data/<split>-*.parquet; naming them keeps STAC Parquet out of splits
    card.data["configs"] = [
        {
            "config_name": "default",
            "data_files": [{"split": name, "path": f"data/{name}-*"} for name in split_counts],
        }
    ]
    return str(card)


def _update_dataset_card(
    dataset_dir: Path,
    records: list[dict],
    split_counts: dict[str, int],
    dataset_version: str,
    sampling: dict | None = None,
    split_stats: dict | None = None,
    splits: SplitAssignment | None = None,
    card_text: CardText | None = None,
) -> None:
    stats = published_stats(records)
    samples_url = None
    (dataset_dir / SAMPLE_TILES_FILENAME).unlink(missing_ok=True)
    if card_text:
        card_path = dataset_dir / CARD_FILENAME
        text = render_card_text(read_card_template(), card_text, dataset_version)
        noun = card_text.label_noun
        samples_url = _write_samples(dataset_dir, records, card_text, dataset_version)
    else:
        card_path = resolve_dataset_card(dataset_dir)
        text = card_path.read_text()
        noun = (_load_dataset_stats(dataset_dir) or {}).get("label_noun", "labels")

    text = _sync_card_dataset_info(text, split_counts)
    generated = GeneratedStats(dataset_dir, stats, noun, dataset_version, samples_url)
    text = _upsert_generated_stats_block(text, generated, sampling, split_stats, splits)

    card_path.write_text(text)
    logger.info("Updated dataset card %s (%d examples)", card_path, stats["tiles"])


def _write_samples(
    dataset_dir: Path, records: list[dict], card_text: CardText, dataset_version: str
) -> str | None:
    """Draw the card's sample tiles; returns their address in the release, or None for none."""
    samples_path = dataset_dir / SAMPLE_TILES_FILENAME
    if not card_text.sample_tiles:
        return None
    feature_type = records[0]["feature_type"]
    if not write_sample_tiles(
        records, feature_type, card_text.sample_tiles, SAMPLE_SEED, samples_path
    ):
        return None
    return (
        f"{HUB_DATASETS_URL}/{card_text.repo_id}/resolve/v{dataset_version}/{SAMPLE_TILES_FILENAME}"
    )


@dataclass(frozen=True)
class GeneratedStats:
    dataset_dir: Path
    published: dict
    noun: str
    dataset_version: str
    samples_url: str | None = None


def _upsert_generated_stats_block(
    card_text: str,
    generated: GeneratedStats,
    sampling: dict | None = None,
    split_stats: dict | None = None,
    splits: SplitAssignment | None = None,
) -> str:
    build_stats = _load_dataset_stats(generated.dataset_dir) or {}
    published = generated.published

    lines = [
        CARD_AUTO_STATS_START,
        "## At a glance",
        "",
        *_glance_lines(generated, build_stats, sampling, split_stats, splits),
        *_sample_lines(generated.samples_url),
        *_class_lines(published["class_counts"], generated.dataset_dir, build_stats),
        "",
        "### Source projects",
        "",
        *source_project_lines(published, generated.noun),
        CARD_AUTO_STATS_END,
    ]
    block = "\n" + "\n".join(lines) + "\n"

    if CARD_AUTO_STATS_START in card_text and CARD_AUTO_STATS_END in card_text:
        pattern = re.compile(
            rf"{re.escape(CARD_AUTO_STATS_START)}.*?{re.escape(CARD_AUTO_STATS_END)}",
            flags=re.DOTALL,
        )
        return pattern.sub(lambda _: "\n".join(lines), card_text, count=1)

    return card_text.rstrip() + "\n" + block


def _glance_lines(
    generated: GeneratedStats,
    build_stats: dict,
    sampling: dict | None,
    split_stats: dict | None,
    splits: SplitAssignment | None,
) -> list[str]:
    published, noun = generated.published, generated.noun
    pixel_sizes = sorted({p["metres_per_pixel"] for p in published["projects"].values()})
    pixel_size = " to ".join(dict.fromkeys(f"{size:.2f} m" for size in pixel_sizes))
    zoom_levels = ", ".join(str(zoom) for zoom in published["zoom_levels"])
    generated_on = str(build_stats.get("generated_at", ""))[:10]
    rows = [
        ("Tiles", f"{published['tiles']:,}"),
        *_split_rows(split_stats, splits),
        (f"Tiles with {noun}", f"{published['tiles_with_labels']:,}"),
        (f"Tiles without {noun}", _empty_tiles(published, sampling)),
        (noun.capitalize(), f"{published['labels']:,}"),
        ("Ground pixel size", f"{pixel_size} (zoom {zoom_levels})"),
        ("Area covered", f"{published['area_sq_km']:,.2f} sq km"),
        ("Countries", ", ".join(published["countries"])),
        (
            "Version",
            generated.dataset_version + (f", built {generated_on}" if generated_on else ""),
        ),
    ]
    return ["| | |", "|---|---|", *(f"| {name} | {value} |" for name, value in rows)]


def _empty_tiles(published: dict, sampling: dict | None) -> str:
    if not sampling:
        return f"{published['tiles_without_labels']:,}"
    return (
        f"{sampling['negatives_kept']:,} of {sampling['negatives_before']:,}, capped at "
        f"{sampling['ratio']:g} per labelled tile in each project"
    )


def _split_rows(
    split_stats: dict[str, dict] | None, splits: SplitAssignment | None
) -> list[tuple[str, str]]:
    if not split_stats or not splits:
        return []
    counts = ", ".join(f"{name} {split_stats[name]['tiles']:,}" for name in SPLIT_NAMES)
    return [("Splits", counts), ("Split method", _split_method(splits))]


def _regions(count: int) -> str:
    return "1 region" if count == 1 else f"{count} regions"


def _split_method(splits: SplitAssignment) -> str:
    settings = splits.settings
    parts = [
        f"{settings.regions_per_project} contiguous regions per project, "
        f"{_regions(settings.test_regions)} to test, "
        f"{_regions(settings.validation_regions)} to validation"
    ]
    excluded = Counter(splits.excluded.values())
    if settings.buffer_meters > 0:
        parts.append(
            f"a {settings.buffer_meters:g} m buffer around held-out regions leaves out "
            f"{excluded[EXCLUDED_BY_BUFFER]:,} tiles"
        )
    else:
        parts.append("tiles at a region edge border tiles of another split")
    if settings.exclude_shared_features:
        parts.append(
            "tiles sharing a feature with a held-out split are left out: "
            f"{excluded[EXCLUDED_BY_SHARED_FEATURE]:,} tiles"
        )
    else:
        parts.append("a feature crossing an edge has parts in both splits")
    return "; ".join(parts)


def _sample_lines(samples_url: str | None) -> list[str]:
    if not samples_url:
        return []
    return [
        "",
        "### Sample tiles",
        "",
        "Each tile is shown beside its labels, coloured by class, with the main class and its "
        "count above.",
        "",
        f"![Sample tiles beside their labels]({samples_url})",
    ]


def _selected_tags(tag: str, build_stats: dict, dataset_dir: Path) -> str:
    values = build_stats.get("tag_values") or ["*"]
    selected = " or ".join(f"`{tag}={value}`" for value in values)
    required = sorted(
        {
            key
            for path in dataset_dir.glob("project_*/metadata.json")
            for key in json.loads(path.read_text()).get("required_tags", [])
        }
    )
    return selected + "".join(f" with `{key}=*`" for key in required)


def _class_lines(class_counts: dict, dataset_dir: Path, build_stats: dict) -> list[str]:
    if not class_counts:
        return []

    tag = build_stats.get("feature_type", "")
    pages = dataset_wiki_pages(dataset_dir)
    wiki = ", ".join(f"[{page}]({url})" for page, url in pages)
    total = sum(class_counts.values())
    lines = [
        "",
        "### Classes",
        "",
        f"Labels are features tagged "
        f"{_selected_tags(tag, build_stats, dataset_dir)}, and a label's class is its `{tag}` "
        "value." + (f" The wiki describes how each tag is mapped: {wiki}." if wiki else ""),
        "",
        "| Class | Labels | Share |",
        "|-------|--------|-------|",
    ]
    lines.extend(
        f"| `{name}` | {count:,} | {100 * count / total:.1f}% |"
        for name, count in list(class_counts.items())[:MAX_CARD_CLASS_ROWS]
    )
    if len(class_counts) > MAX_CARD_CLASS_ROWS:
        lines.append(f"| {len(class_counts) - MAX_CARD_CLASS_ROWS} more classes | | |")
    return lines


def _load_dataset_stats(dataset_dir: Path) -> dict | None:
    stats_path = dataset_dir / "dataset_stats.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())
