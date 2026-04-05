import json
import logging
import re
import tarfile
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from huggingface_hub import DatasetCard, HfApi

logger = logging.getLogger(__name__)

TILE_ID_PATTERN = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)$")
TILE_TITLE_PATTERN = re.compile(r"Tile\(x=(\d+), y=(\d+), z=(\d+)\)")
CARD_AUTO_STATS_START = "<!-- AUTO_STATS_START -->"
CARD_AUTO_STATS_END = "<!-- AUTO_STATS_END -->"


def generate_parquet(
    dataset_dir: Path, output_dir: Path | None = None, dataset_version: str = "1.0.0"
):
    output_dir = output_dir or dataset_dir / "parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for project_dir in sorted(dataset_dir.glob("project_*")):
        metadata_path = project_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        project_id = metadata["project_id"]
        project_name = metadata["name"]
        country = metadata.get("country", "")
        organisation = metadata.get("organisation", "")
        imagery_url = metadata.get("imagery_url", "")

        chips_dir = project_dir / "chips"
        masks_dir = project_dir / "masks"
        labels_dir = project_dir / "labels"
        tiles_geojson_path = project_dir / "tiles.geojson"

        tile_bounds = _load_tile_bounds(tiles_geojson_path)

        for chip_path in sorted(chips_dir.glob("OAM-*.tif")):
            tile_id = chip_path.stem
            label_path = labels_dir / f"{tile_id}.geojson"
            mask_path = masks_dir / f"{tile_id}.tif"

            if not label_path.exists():
                continue

            tile_match = TILE_ID_PATTERN.match(tile_id)
            if not tile_match:
                continue

            tile_x, tile_y, tile_z = (
                int(tile_match.group(1)),
                int(tile_match.group(2)),
                int(tile_match.group(3)),
            )

            label_data = json.loads(label_path.read_text())
            num_buildings = len(label_data.get("features", []))
            label_geojson = json.dumps(label_data)

            bounds = tile_bounds.get(tile_id, {})

            records.append(
                {
                    "image": str(chip_path),
                    "mask": str(mask_path) if mask_path.exists() else None,
                    "tile_id": tile_id,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_z": tile_z,
                    "project_id": project_id,
                    "project_name": project_name,
                    "country": country,
                    "organisation": organisation,
                    "imagery_url": imagery_url,
                    "num_buildings": num_buildings,
                    "label_geojson": label_geojson,
                    "bbox_west": bounds.get("west", 0.0),
                    "bbox_south": bounds.get("south", 0.0),
                    "bbox_east": bounds.get("east", 0.0),
                    "bbox_north": bounds.get("north", 0.0),
                }
            )

    if not records:
        logger.warning("No valid tile pairs found for Parquet generation")
        return

    logger.info("Generating Parquet with %d tile records", len(records))

    features = Features(
        {
            "image": Image(),
            "mask": Image(),
            "tile_id": Value("string"),
            "tile_x": Value("int32"),
            "tile_y": Value("int32"),
            "tile_z": Value("int32"),
            "project_id": Value("int32"),
            "project_name": Value("string"),
            "country": Value("string"),
            "organisation": Value("string"),
            "imagery_url": Value("string"),
            "num_buildings": Value("int32"),
            "label_geojson": Value("large_string"),
            "bbox_west": Value("float64"),
            "bbox_south": Value("float64"),
            "bbox_east": Value("float64"),
            "bbox_north": Value("float64"),
        }
    )

    column_data = {key: [r[key] for r in records] for key in records[0]}

    dataset = Dataset.from_dict(column_data, features=features)
    dataset.info.version = dataset_version
    dataset.to_parquet(str(output_dir / "data.parquet"))

    _update_dataset_card(dataset_dir, num_examples=len(records), dataset_version=dataset_version)

    logger.info("Parquet saved to %s (version %s)", output_dir, dataset_version)


def push_to_hub(
    dataset_dir: Path,
    repo_id: str,
    parquet_dir: Path | None = None,
    dataset_version: str = "1.0.0",
):
    parquet_dir = parquet_dir or dataset_dir / "parquet"
    parquet_path = parquet_dir / "data.parquet"

    if not parquet_path.exists():
        logger.info("Parquet not found, generating first")
        generate_parquet(dataset_dir, parquet_dir, dataset_version=dataset_version)

    logger.info("Loading Parquet dataset from %s", parquet_path)
    dataset = Dataset.from_parquet(str(parquet_path))

    _update_dataset_card(dataset_dir, num_examples=len(dataset), dataset_version=dataset_version)

    logger.info("Pushing to HuggingFace Hub: %s", repo_id)
    _push_dataset_with_card_retry(dataset=dataset, repo_id=repo_id, dataset_version=dataset_version)

    dataset_card_path = Path("dataset_card.md")
    if dataset_card_path.exists():
        logger.info("Uploading dataset card")
        card = DatasetCard(dataset_card_path.read_text())
        card.push_to_hub(repo_id, commit_message=f"Update dataset card v{dataset_version}")

    _tag_hf_revision(repo_id, dataset_version)
    logger.info("Dataset published to https://huggingface.co/datasets/%s", repo_id)


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
    ignore = ["parquet/*", ".cache/*"] + root_level_files
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

    logger.info("Raw dataset uploaded to https://huggingface.co/datasets/%s/tree/main/raw", repo_id)


def _tag_hf_revision(repo_id: str, dataset_version: str) -> None:
    tag = f"v{dataset_version}"
    api = HfApi()
    try:
        api.create_tag(repo_id=repo_id, repo_type="dataset", tag=tag, tag_message=tag)
        logger.info("Tagged HF dataset revision: %s", tag)
    except Exception as exc:
        logger.warning("Could not create HF tag %s: %s", tag, exc)


def _push_dataset_with_card_retry(dataset: Dataset, repo_id: str, dataset_version: str) -> None:
    commit_message = f"v{dataset_version}"

    try:
        dataset.push_to_hub(repo_id, max_shard_size="500MB", commit_message=commit_message)
        return
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

    api = HfApi()
    try:
        api.delete_file(
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Reset README to recover from metadata merge bug",
        )
    except Exception as delete_exc:
        logger.warning("Could not delete remote README before retry: %s", delete_exc)

    dataset.push_to_hub(repo_id, max_shard_size="500MB", commit_message=commit_message)


def _load_tile_bounds(tiles_geojson_path: Path) -> dict[str, dict]:
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
        tile_id = f"OAM-{match.group(1)}-{match.group(2)}-{match.group(3)}"

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


def _update_dataset_card(
    dataset_dir: Path,
    num_examples: int,
    dataset_version: str,
) -> None:
    dataset_card_path = Path("dataset_card.md")
    if not dataset_card_path.exists():
        return

    card_text = dataset_card_path.read_text()
    card_text = _upsert_generated_stats_block(card_text, dataset_dir, num_examples, dataset_version)

    dataset_card_path.write_text(card_text)
    logger.info("Updated dataset card dynamic stats (%d examples)", num_examples)


def _upsert_generated_stats_block(
    card_text: str,
    dataset_dir: Path,
    num_examples: int,
    dataset_version: str,
) -> str:
    stats = _load_dataset_stats(dataset_dir)

    lines = [
        CARD_AUTO_STATS_START,
        "## Generated Stats",
        f"- Dataset version: {dataset_version}",
        f"- Tile pairs (num_examples): {num_examples:,}",
    ]

    if stats:
        lines.extend(
            [
                f"- Total projects: {stats.get('total_projects', 0):,}",
                f"- Tiles with features: {stats.get('tiles_with_buildings', 0):,}",
                f"- Tiles without features: {stats.get('tiles_without_buildings', 0):,}",
                f"- Total polygons: {stats.get('total_building_polygons', 0):,}",
                f"- Avg buildings per tile: {stats.get('avg_buildings_per_tile', 0)}",
                f"- Total area: ~{stats.get('total_area_sq_km', 0):,.1f} sq km",
                f"- Countries: {len(stats.get('countries', {}))}",
                f"- Generated at: {stats.get('generated_at', '')}",
            ]
        )

        country_details = stats.get("country_details", {})
        if country_details:
            lines.append("")
            lines.append("### Coverage by Country")
            lines.append("")
            lines.append("| Country | Projects | Tiles | Buildings | Area (sq km) |")
            lines.append("|---------|----------|-------|-----------|-------------|")
            for country, info in country_details.items():
                lines.append(
                    f"| {country} | {info['projects']} "
                    f"| {info['tiles']:,} | {info['polygons']:,} "
                    f"| {info['area_sq_km']:,.1f} |"
                )

    lines.append(CARD_AUTO_STATS_END)
    block = "\n" + "\n".join(lines) + "\n"

    if CARD_AUTO_STATS_START in card_text and CARD_AUTO_STATS_END in card_text:
        pattern = re.compile(
            rf"{re.escape(CARD_AUTO_STATS_START)}.*?{re.escape(CARD_AUTO_STATS_END)}",
            flags=re.DOTALL,
        )
        return pattern.sub("\n".join(lines), card_text, count=1)

    return card_text.rstrip() + "\n" + block


def _load_dataset_stats(dataset_dir: Path) -> dict | None:
    stats_path = dataset_dir / "dataset_stats.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())
