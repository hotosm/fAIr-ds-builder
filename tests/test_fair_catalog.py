import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pystac
import pytest

from hot_oam_dataset import fair_catalog
from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.fair_catalog import FairSettings, build_fair_dataset_item, publish_to_fair

pytest.importorskip("fair.stac.builders", reason="fAIr publishing needs the fair extra")
from fair.stac.constants import DATASETS_COLLECTION
from fair.stac.validators import validate_item

CARD = CardText.from_mapping(
    {
        "repo_id": "hotosm/fAIr-guru-td-trees",
        "title": "fAIr Open Mapping Gurus: trees",
        "label_noun": "trees",
        "tasks": ["detection"],
        "summary": "Tree training data from Freetown.",
        "selection": "Trees mapped as points and tree groups as polygons.",
        "limitations": [],
    }
)
SETTINGS = FairSettings(user_id="hot", provider="hot")


@pytest.fixture
def dataset_dir(tmp_path):
    folder = tmp_path / "fAIr-guru-td-trees"
    (folder / "parquet").mkdir(parents=True)
    for split, rows in (("train", 8), ("validation", 1), ("test", 1)):
        table = pa.table({"tile_id": [f"OAM-{i}-0-20" for i in range(rows)]})
        pq.write_table(table, folder / "parquet" / f"{split}.parquet")
    (folder / "dataset_stats.json").write_text(
        json.dumps({"feature_type": "natural", "tag_values": ["tree", "tree_group"]})
    )
    (folder / "project_50661").mkdir()
    (folder / "project_50661" / "metadata.json").write_text(
        json.dumps({"geometry_types": ["point", "polygon"]})
    )
    collection = pystac.Collection(
        id="fAIr-guru-td-trees",
        description="Trees",
        extent=pystac.Extent(
            pystac.SpatialExtent([[-13.3, 8.4, -13.1, 8.5]]),
            pystac.TemporalExtent([[datetime(2026, 10, 17, tzinfo=UTC), None]]),
        ),
    )
    collection.normalize_hrefs(str(folder / "stac"))
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return folder


def test_dataset_item_passes_fair_validation_and_points_at_the_release(dataset_dir):
    item = build_fair_dataset_item(dataset_dir, CARD, SETTINGS, "0.1.0")

    assert validate_item(item) == []
    assert item.id == "fAIr-guru-td-trees"
    release = "hf://datasets/hotosm/fAIr-guru-td-trees@v0.1.0/data"
    assert item.assets["chips"].href == release
    assert item.assets["labels"].href == release
    assert item.assets["chips"].media_type == "application/vnd.apache.parquet"
    assert item.properties["fair:user_id"] == "hot"
    assert item.properties["fair:chip_count"] == 10
    assert item.properties["providers"] == [{"name": "hot", "roles": ["producer", "processor"]}]
    assert item.properties["label:classes"] == [
        {"name": "natural", "classes": ["tree", "tree_group"]}
    ]
    assert item.properties["label:tasks"] == ["detection"]
    assert set(item.properties["keywords"]) == {"natural", "point", "polygon"}
    assert item.bbox == [-13.3, 8.4, -13.1, 8.5]


def test_publish_writes_the_item_to_the_datasets_collection(dataset_dir, monkeypatch, mocker):
    monkeypatch.setenv(fair_catalog.FAIR_API_KEY_ENV, "key")
    mocker.patch.object(fair_catalog, "release_tags", return_value={"v0.1.0"})
    backend = mocker.patch("fair.stac.api_backend.StacApiBackend")

    publish_to_fair(dataset_dir, "https://stac.example.org/stac", CARD, SETTINGS, "0.1.0")

    backend.assert_called_once_with("https://stac.example.org/stac", api_key="key")
    collection_id, item = backend.return_value.publish_item.call_args.args
    assert collection_id == DATASETS_COLLECTION
    assert item.id == "fAIr-guru-td-trees"


def test_publish_refuses_without_an_api_key(dataset_dir, monkeypatch):
    monkeypatch.delenv(fair_catalog.FAIR_API_KEY_ENV, raising=False)

    with pytest.raises(ValueError, match="FAIR_STAC_API_KEY"):
        publish_to_fair(dataset_dir, "https://stac.example.org/stac", CARD, SETTINGS, "0.1.0")


def test_publish_refuses_a_version_not_released_on_the_hub(dataset_dir, monkeypatch, mocker):
    monkeypatch.setenv(fair_catalog.FAIR_API_KEY_ENV, "key")
    mocker.patch.object(fair_catalog, "release_tags", return_value=set())
    backend = mocker.patch("fair.stac.api_backend.StacApiBackend")

    with pytest.raises(ValueError, match=r"has no tag v0\.1\.0"):
        publish_to_fair(dataset_dir, "https://stac.example.org/stac", CARD, SETTINGS, "0.1.0")
    backend.assert_not_called()


@pytest.mark.parametrize(
    ("mapping", "message"),
    [({"user_id": "hot"}, "exactly"), ({"user_id": " ", "provider": "hot"}, "must not be empty")],
)
def test_invalid_fair_settings_are_rejected(mapping, message):
    with pytest.raises(ValueError, match=message):
        FairSettings.from_mapping(mapping)


def test_publish_refuses_a_plain_http_api(dataset_dir, monkeypatch, mocker):
    monkeypatch.setenv(fair_catalog.FAIR_API_KEY_ENV, "key")
    backend = mocker.patch("fair.stac.api_backend.StacApiBackend")

    with pytest.raises(ValueError, match="must use https"):
        publish_to_fair(dataset_dir, "http://stac.example.org/stac", CARD, SETTINGS, "0.1.0")
    backend.assert_not_called()
