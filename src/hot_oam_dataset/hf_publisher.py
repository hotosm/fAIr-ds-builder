import json
import logging
import re
from pathlib import Path

from datasets import Dataset, Features, Image, Value
from huggingface_hub import DatasetCard, HfApi

logger = logging.getLogger(__name__)

TILE_ID_PATTERN = re.compile(r"^OAM-(\d+)-(\d+)-(\d+)$")
TILE_TITLE_PATTERN = re.compile(r"Tile\(x=(\d+), y=(\d+), z=(\d+)\)")


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
        labels_dir = project_dir / "labels"
        tiles_geojson_path = project_dir / "tiles.geojson"

        tile_bounds = _load_tile_bounds(tiles_geojson_path)

        for chip_path in sorted(chips_dir.glob("OAM-*.tif")):
            tile_id = chip_path.stem
            label_path = labels_dir / f"{tile_id}.geojson"

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

    logger.info("Pushing to HuggingFace Hub: %s", repo_id)
    dataset.push_to_hub(repo_id, max_shard_size="500MB", commit_message=f"v{dataset_version}")

    dataset_card_path = Path("dataset_card.md")
    if dataset_card_path.exists():
        logger.info("Uploading dataset card")
        card = DatasetCard(dataset_card_path.read_text())
        card.push_to_hub(repo_id, commit_message=f"Update dataset card v{dataset_version}")

    logger.info("Dataset published to https://huggingface.co/datasets/%s", repo_id)


def push_raw_to_hub(dataset_dir: Path, repo_id: str, dataset_version: str = "1.0.0"):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.create_branch(repo_id=repo_id, repo_type="dataset", branch="raw", exist_ok=True)

    logger.info("Uploading raw dataset folder to %s@raw (large folder mode)", repo_id)
    api.upload_large_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        revision="raw",
        ignore_patterns=["parquet/*", ".cache/*"],
    )
    logger.info(
        "Raw dataset uploaded to https://huggingface.co/datasets/%s/tree/raw", repo_id
    )


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
