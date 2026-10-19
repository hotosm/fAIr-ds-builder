"""Register a published dataset in fAIr's STAC catalogue as one dataset item."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import pystac
from shapely.geometry import box, mapping

from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.hf_publisher import HUB_DATASETS_URL, release_tags
from hot_oam_dataset.split_generator import SPLIT_NAMES
from hot_oam_dataset.stac_export import SPLIT_MAP_FILENAME, STAC_DIRNAME

logger = logging.getLogger(__name__)

FAIR_API_KEY_ENV = "FAIR_STAC_API_KEY"
DATASET_LICENSE = "ODbL-1.0"
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
GEOMETRY_TYPES = {"point", "line", "polygon"}


@dataclass(frozen=True)
class FairSettings:
    """Who registers the dataset in fAIr; read from the `fair` key of a dataset config."""

    user_id: str
    provider: str

    @classmethod
    def from_mapping(cls, mapping: dict) -> "FairSettings":
        expected = {"user_id", "provider"}
        if set(mapping) != expected:
            raise ValueError(f"fair needs exactly {sorted(expected)}, got {sorted(mapping)}")
        empty = sorted(key for key in expected if not str(mapping[key]).strip())
        if empty:
            raise ValueError(f"fair fields must not be empty: {empty}")
        return cls(
            user_id=str(mapping["user_id"]).strip(), provider=str(mapping["provider"]).strip()
        )


def _release_href(repo_id: str, dataset_version: str, path: str) -> str:
    """An fsspec path into the dataset's release tag, so the item never points at moving files."""
    return f"hf://datasets/{repo_id}@v{dataset_version}/{path}"


def _published_tiles(dataset_dir: Path) -> int:
    parquet_dir = dataset_dir / "parquet"
    return sum(pq.read_metadata(parquet_dir / f"{name}.parquet").num_rows for name in SPLIT_NAMES)


def _stats(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "dataset_stats.json").read_text())


def build_fair_dataset_item(
    dataset_dir: Path, card_text: CardText, settings: FairSettings, dataset_version: str
) -> pystac.Item:
    """The dataset as a fAIr dataset item; chips and labels point at the release's Parquet."""
    # fair-py-ops is an optional extra: it brings zenml, mlflow and onnxruntime
    from fair.stac.builders import build_dataset_item

    stats = _stats(dataset_dir)
    collection = pystac.Collection.from_file(str(dataset_dir / STAC_DIRNAME / "collection.json"))
    bbox = collection.extent.spatial.bboxes[0]
    geometry_types = sorted(
        {
            kind
            for metadata_path in dataset_dir.glob("project_*/metadata.json")
            for kind in json.loads(metadata_path.read_text())["geometry_types"]
        }
    )
    feature_type = stats["feature_type"]
    tag_values = stats["tag_values"] or ["*"]
    data_href = _release_href(card_text.repo_id, dataset_version, "data")
    hub_url = f"{HUB_DATASETS_URL}/{card_text.repo_id}"

    item = build_dataset_item(
        label_type="vector",
        label_tasks=list(card_text.tasks),
        label_classes=[{"name": feature_type, "classes": tag_values}],
        keywords=[feature_type, *geometry_types],
        chips_href=data_href,
        labels_href=data_href,
        title=card_text.title,
        description=card_text.summary,
        user_id=settings.user_id,
        providers=[{"name": settings.provider, "roles": ["producer", "processor"]}],
        item_id=collection.id,
        download_href=f"{hub_url}/tree/v{dataset_version}",
        thumbnail_href=f"{hub_url}/resolve/v{dataset_version}/{STAC_DIRNAME}/{SPLIT_MAP_FILENAME}",
        chip_count=_published_tiles(dataset_dir),
        geometry=mapping(box(*bbox)),
        bbox=list(bbox),
        version=dataset_version,
        license_id=DATASET_LICENSE,
        label_description=card_text.selection,
    )
    # build_dataset_item assumes a chips folder and a GeoJSON file; these point at the Parquet
    item.assets["chips"].media_type = PARQUET_MEDIA_TYPE
    item.assets["labels"].media_type = PARQUET_MEDIA_TYPE
    item.assets["download"].media_type = pystac.MediaType.HTML
    return item


def _require_release_tag(repo_id: str, dataset_version: str) -> None:
    if f"v{dataset_version}" not in release_tags(repo_id):
        msg = f"{repo_id} has no tag v{dataset_version}; run --push-to-hub with this version first"
        raise ValueError(msg)


def publish_to_fair(
    dataset_dir: Path,
    stac_api_url: str,
    card_text: CardText,
    settings: FairSettings,
    dataset_version: str,
) -> pystac.Item:
    """Write the dataset item to fAIr's STAC API, pointing at the dataset's Hub release."""
    from fair.stac.api_backend import StacApiBackend
    from fair.stac.constants import DATASETS_COLLECTION
    from fair.stac.validators import validate_item

    if urlsplit(stac_api_url).scheme != "https":
        msg = f"The STAC API URL must use https, as the key is sent to it: {stac_api_url}"
        raise ValueError(msg)
    api_key = os.environ.get(FAIR_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Set {FAIR_API_KEY_ENV} to a fAIr STAC API key to publish")
    _require_release_tag(card_text.repo_id, dataset_version)

    item = build_fair_dataset_item(dataset_dir, card_text, settings, dataset_version)
    errors = validate_item(item)
    if errors:
        raise ValueError(f"The fAIr dataset item is invalid: {errors}")
    backend = StacApiBackend(stac_api_url, api_key=api_key)
    published = backend.publish_item(DATASETS_COLLECTION, item)
    logger.info("Published %s to %s/collections/%s", item.id, stac_api_url, DATASETS_COLLECTION)
    return published
