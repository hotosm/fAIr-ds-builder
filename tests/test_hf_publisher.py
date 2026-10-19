import collections
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pystac
import pytest
import rasterio
from huggingface_hub import DatasetCard
from huggingface_hub.errors import HfHubHTTPError
from rasterio.transform import from_origin

from hot_oam_dataset import hf_publisher
from hot_oam_dataset.dataset_card import CardText, published_stats
from hot_oam_dataset.hf_publisher import (
    GeneratedStats,
    _cap_negatives,
    _collect_parquet_records,
    _parquet_features,
    _update_dataset_card,
    _upsert_generated_stats_block,
    generate_parquet,
    resolve_dataset_card,
    split_stats_from_records,
)
from hot_oam_dataset.split_generator import SplitAssignment, SplitSettings

CARD_TEMPLATE = """\
# Dataset Card
Some content here.

<!-- AUTO_STATS_START -->
<!-- AUTO_STATS_END -->
"""

CARD_TEMPLATE_WITH_METADATA = """\
---
pretty_name: Test Dataset
dataset_info:
  features:
    - name: image
      dtype: image
  splits:
    - name: train
      num_examples: 57890
    - name: validation
      num_examples: 7237
    - name: test
      num_examples: 7236
---

# Dataset Card

<!-- AUTO_STATS_START -->
<!-- AUTO_STATS_END -->
"""


def _write_dataset_stats(dataset_dir, projects, country_details=None):
    stats = {
        "total_projects": len(projects),
        "tiles_with_features": sum(p["valid_pairs"] for p in projects),
        "tiles_without_features": 0,
        "total_label_features": sum(p.get("total_label_features", 0) for p in projects),
        "avg_features_per_tile": 5.0,
        "total_area_sq_km": 100.0,
        "countries": {},
        "country_details": country_details or {},
        "generated_at": "2026-04-06T00:00:00+00:00",
        "projects": projects,
    }
    (dataset_dir / "dataset_stats.json").write_text(json.dumps(stats))


@pytest.fixture
def chip(tmp_path):
    """A small RGB chip for test rows to point at, read by the brightness estimate."""
    path = tmp_path / "chip.tif"
    if not path.exists():
        profile = {"driver": "GTiff", "width": 16, "height": 16, "count": 3, "dtype": "uint8"}
        pixel = 0.0007 / 16
        profile |= {"crs": "EPSG:4326", "transform": from_origin(0, 0.0007, pixel, pixel)}
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(np.full((3, 16, 16), 120, dtype=np.uint8))
    return str(path)


def _row(image, project_id=1, country="Ghana", num_features=1, tile_id="OAM-1-1-19"):
    return {
        "image": image,
        "tile_id": tile_id,
        "project_id": project_id,
        "project_name": f"Project {project_id}",
        "project_url": f"https://tasks.hotosm.org/projects/{project_id}",
        "country": country,
        "num_features": num_features,
        "class_counts": json.dumps({"yes": num_features} if num_features else {}),
        "tile_z": 19,
        "bbox_west": 0.0,
        "bbox_south": 0.0,
        "bbox_east": 0.0007,
        "bbox_north": 0.0007,
    }


def _generated(dataset_dir, rows, version="0.2.0"):
    return GeneratedStats(dataset_dir, published_stats(rows), "labels", version)


@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path


@pytest.fixture
def output_dir(tmp_path, monkeypatch):
    """A dataset folder, run from an empty working directory as an installed package would be."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    folder = tmp_path / "freetown-dataset"
    folder.mkdir()
    return folder


def test_new_dataset_card_is_seeded_from_the_packaged_template(output_dir, chip):
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )

    _update_dataset_card(
        output_dir, [_row(chip)] * 10, split_counts={"train": 10}, dataset_version="0.3.0"
    )

    card_text = (output_dir / "dataset_card.md").read_text()
    assert "<!-- DESCRIPTION_START -->" in card_text
    assert "## Load" in card_text
    assert "| Version | 0.3.0" in card_text
    assert "| Tiles | 10 |" in card_text
    assert "<selection>" not in card_text
    assert "## Known limitations" not in card_text


def test_card_front_matter_splits_match_published_counts(output_dir, chip):
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )

    _update_dataset_card(
        output_dir,
        [_row(chip)] * 10,
        split_counts={"train": 8, "validation": 1, "test": 1},
        dataset_version="0.3.0",
    )

    card = DatasetCard((output_dir / "dataset_card.md").read_text())
    assert card.data["configs"] == [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": "data/train-*"},
                {"split": "validation", "path": "data/validation-*"},
                {"split": "test", "path": "data/test-*"},
            ],
        }
    ]
    assert card.data["dataset_info"]["splits"] == [
        {"name": "train", "num_examples": 8},
        {"name": "validation", "num_examples": 1},
        {"name": "test", "num_examples": 1},
    ]


def test_card_states_the_split_buffer_and_tiles_it_left_out(output_dir, chip):
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )
    rows = {"train": [_split_row(1, "Ghana", 5)] * 8, "validation": [], "test": []}

    _update_dataset_card(
        output_dir,
        [_row(chip)] * 8,
        split_counts={"train": 8},
        dataset_version="0.3.0",
        split_stats=split_stats_from_records(rows),
        splits=SplitAssignment(
            SplitSettings(buffer_meters=150),
            tiles={},
            excluded=dict.fromkeys(["OAM-1-1-19", "OAM-1-2-19"], "buffer"),
        ),
    )

    card_text = (output_dir / "dataset_card.md").read_text()
    assert "| Split method | 10 contiguous regions per project" in card_text
    assert "a 150 m buffer around held-out regions leaves out 2 tiles" in card_text


def test_existing_dataset_card_is_not_reseeded(output_dir):
    own_card = output_dir / "dataset_card.md"
    own_card.write_text("# Freetown card\n")

    assert resolve_dataset_card(output_dir) == own_card
    assert own_card.read_text() == "# Freetown card\n"


def test_stats_block_includes_split_table(dataset_dir, chip):
    projects = [
        {"project_id": 1, "valid_pairs": 800, "total_label_features": 5000, "country": "Kenya"},
        {"project_id": 2, "valid_pairs": 100, "total_label_features": 600, "country": "Peru"},
        {"project_id": 3, "valid_pairs": 100, "total_label_features": 400, "country": "Ghana"},
    ]
    _write_dataset_stats(dataset_dir, projects)
    split_stats = split_stats_from_records(
        {
            "train": [_split_row(1, "Kenya", 5000)] * 800,
            "validation": [_split_row(2, "Peru", 600)] * 100,
            "test": [_split_row(3, "Ghana", 400)] * 100,
        }
    )

    result = _upsert_generated_stats_block(
        CARD_TEMPLATE,
        _generated(dataset_dir, [_row(chip)]),
        split_stats=split_stats,
        splits=SplitAssignment(SplitSettings(), tiles={}, excluded={}),
    )

    assert "| Splits | train " in result
    assert ", validation " in result
    assert ", test " in result
    assert "tiles at a region edge border tiles of another split" in result
    assert "a feature crossing an edge has parts in both splits" in result


def test_card_counts_the_published_rows_not_the_raw_build(output_dir, chip):
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 48490, "total_label_features": 3927, "country": "X"}],
    )
    rows = [_row(chip, num_features=2)] * 3 + [_row(chip, num_features=0)]
    sampling = {"ratio": 1.0, "negatives_kept": 1, "negatives_before": 46555}

    _update_dataset_card(
        output_dir, rows, split_counts={"train": 4}, dataset_version="0.1.0", sampling=sampling
    )

    card_text = (output_dir / "dataset_card.md").read_text()
    assert "| Tiles | 4 |" in card_text
    assert "| Tiles without labels | 1 of 46,555, capped at 1 per labelled tile" in card_text
    assert "| Ghana | 4 | 6 |" in card_text


@pytest.mark.parametrize("sample_tiles", [6, 0])
def test_config_card_shows_sample_tiles_from_the_release(output_dir, chip, sample_tiles):
    card_text = CardText.from_mapping(
        {
            "repo_id": "hotosm/test",
            "title": "Test",
            "label_noun": "buildings",
            "tasks": ["segmentation"],
            "summary": "Summary.",
            "selection": "Selection.",
            "limitations": [],
            "sample_tiles": sample_tiles,
        }
    )
    building = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0.0003, 0.0003]},
        "properties": {"tags": {"building": "yes"}},
    }
    row = _row(chip) | {
        "feature_type": "building",
        "label_geojson": json.dumps({"type": "FeatureCollection", "features": [building]}),
    }

    _update_dataset_card(
        output_dir, [row], split_counts={"train": 1}, dataset_version="0.1.0", card_text=card_text
    )

    card = (output_dir / "dataset_card.md").read_text()
    url = "https://huggingface.co/datasets/hotosm/test/resolve/v0.1.0/samples.png"
    assert (f"![Sample tiles beside their labels]({url})" in card) == bool(sample_tiles)
    assert (output_dir / "samples.png").exists() == bool(sample_tiles)


def _split_row(project_id, country, features):
    return {"project_id": project_id, "country": country, "num_features": features}


@pytest.mark.parametrize(
    ("tag_values", "expected"),
    [(["tree", "tree_group"], "`natural=tree` or `natural=tree_group`"), ([], "`natural=*`")],
)
def test_stats_block_names_the_tag_values(dataset_dir, tag_values, expected, chip):
    _write_dataset_stats(
        dataset_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Kenya"}],
    )
    stats_path = dataset_dir / "dataset_stats.json"
    stats = json.loads(stats_path.read_text())
    stats |= {"feature_type": "natural", "tag_values": tag_values}
    stats_path.write_text(json.dumps(stats))

    result = _upsert_generated_stats_block(CARD_TEMPLATE, _generated(dataset_dir, [_row(chip)]))

    assert f"Labels are OpenStreetMap features tagged {expected}, and a label" in result


def test_stats_block_names_the_tags_every_label_also_carries(dataset_dir, chip):
    _write_dataset_stats(
        dataset_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Kenya"}],
    )
    stats_path = dataset_dir / "dataset_stats.json"
    stats = json.loads(stats_path.read_text()) | {"feature_type": "surface"}
    stats_path.write_text(json.dumps(stats))
    (dataset_dir / "project_1").mkdir()
    metadata = {"feature_type": "surface", "tag_values": [], "required_tags": ["highway"]}
    (dataset_dir / "project_1" / "metadata.json").write_text(json.dumps(metadata))

    result = _upsert_generated_stats_block(CARD_TEMPLATE, _generated(dataset_dir, [_row(chip)]))

    assert "features tagged `surface=*` with `highway=*`, and a label" in result


def test_stats_block_omits_splits_when_no_splits_file(dataset_dir, chip):
    projects = [
        {"project_id": 1, "valid_pairs": 100, "total_label_features": 50, "country": "Kenya"},
    ]
    _write_dataset_stats(dataset_dir, projects)
    result = _upsert_generated_stats_block(CARD_TEMPLATE, _generated(dataset_dir, [_row(chip)]))

    assert "| Splits |" not in result
    assert "<!-- AUTO_STATS_START -->" in result
    assert "<!-- AUTO_STATS_END -->" in result


def _write_tile_project(dataset_dir, tile_prefix, feature_type, extra_metadata=None):
    project_dir = dataset_dir / "project_9"
    (project_dir / "chips").mkdir(parents=True)
    (project_dir / "labels").mkdir()

    metadata = {
        "project_id": 9,
        "name": "Road mapping",
        "country": "Nepal",
        "organisation": "HOT",
        "imagery_url": "https://tiles.openaerialmap.org/x/{z}/{x}/{y}",
        "feature_type": feature_type,
    }
    metadata.update(extra_metadata or {})
    (project_dir / "metadata.json").write_text(json.dumps(metadata))

    tile_id = f"{tile_prefix}-100-200-19"
    (project_dir / "chips" / f"{tile_id}.tif").write_bytes(b"chip")
    (project_dir / "labels" / f"{tile_id}.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": None}],
            }
        )
    )
    (project_dir / "tiles.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"title": "Tile(x=100, y=200, z=19)"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                        },
                    }
                ],
            }
        )
    )
    return tile_id


def test_records_follow_the_project_tile_prefix(dataset_dir):
    tile_id = _write_tile_project(dataset_dir, "TMS", "highway", {"tile_prefix": "TMS"})

    records = _collect_parquet_records(dataset_dir)

    assert len(records) == 1
    record = records[0]
    assert record["tile_id"] == tile_id
    assert (record["tile_x"], record["tile_y"], record["tile_z"]) == (100, 200, 19)
    assert record["feature_type"] == "highway"
    assert record["num_features"] == 1
    assert record["bbox_east"] == 1.0


def test_records_default_to_the_oam_prefix(dataset_dir):
    _write_tile_project(dataset_dir, "OAM", "building")

    records = _collect_parquet_records(dataset_dir)

    assert [r["tile_id"] for r in records] == ["OAM-100-200-19"]
    assert records[0]["feature_type"] == "building"


def test_card_schema_follows_the_parquet_columns(output_dir, chip):
    """A card declaring different columns than the Parquet makes load_dataset fail."""
    (output_dir / "dataset_card.md").write_text(CARD_TEMPLATE_WITH_METADATA)
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )

    _update_dataset_card(
        output_dir, [_row(chip)] * 10, split_counts={"train": 10}, dataset_version="0.3.0"
    )

    card = DatasetCard((output_dir / "dataset_card.md").read_text())
    declared = [f["name"] for f in card.data["dataset_info"]["features"]]
    assert declared == list(_parquet_features())


def _rec(project_id, tile, n_features):
    return {"project_id": project_id, "tile_id": tile, "num_features": n_features}


def test_cap_negatives_keeps_every_labelled_tile():
    records = [_rec(1, f"OAM-{i}-0-19", i % 3) for i in range(30)]
    positives = [r for r in records if r["num_features"] > 0]

    kept, summary = _cap_negatives(records, 1.0)

    assert all(p in kept for p in positives)
    assert summary["negatives_kept"] <= len(positives)


def test_cap_negatives_applies_ratio_per_project():
    records = [_rec(1, f"OAM-{i}-0-19", 1) for i in range(10)]
    records += [_rec(1, f"OAM-1{i}-0-19", 0) for i in range(50)]
    records += [_rec(2, f"OAM-2{i}-0-19", 1) for i in range(2)]
    records += [_rec(2, f"OAM-3{i}-0-19", 0) for i in range(40)]

    kept, _ = _cap_negatives(records, 1.0)

    per_project = collections.Counter(r["project_id"] for r in kept)
    assert per_project[1] == 20
    assert per_project[2] == 4


def test_cap_negatives_keeps_all_when_negatives_are_scarce():
    records = [_rec(1, f"OAM-{i}-0-19", 1) for i in range(10)] + [_rec(1, "OAM-99-0-19", 0)]

    kept, summary = _cap_negatives(records, 1.0)

    assert len(kept) == 11
    assert summary["negatives_kept"] == 1


def test_cap_negatives_spreads_the_sample():
    records = [_rec(1, "OAM-0-0-19", 1)] + [_rec(1, f"OAM-{i:03d}-0-19", 0) for i in range(1, 101)]

    kept, _ = _cap_negatives(records, 5.0)
    tiles = sorted(r["tile_id"] for r in kept if r["num_features"] == 0)

    assert len(tiles) == 5
    assert tiles[0] != tiles[-1]
    assert len(set(tiles)) == 5


def _small_catalogue(stac_dir):
    item = pystac.Item(
        "project-1",
        geometry={"type": "Point", "coordinates": [0, 0]},
        bbox=[0, 0, 0, 0],
        datetime=datetime(2026, 10, 17, tzinfo=UTC),
        properties={},
    )
    item.add_asset("imagery", pystac.Asset(href=str(stac_dir / "project-1" / "imagery.tif")))
    collection = pystac.Collection(
        "test", "Test", pystac.Extent.from_items([item]), license="ODbL-1.0"
    )
    collection.add_item(item)
    collection.normalize_hrefs(str(stac_dir))
    (stac_dir / "project-1").mkdir(parents=True)
    (stac_dir / "project-1" / "imagery.tif").write_bytes(b"cog")
    collection.make_all_asset_hrefs_relative()
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)


def test_push_uploads_the_stac_with_links_pinned_to_the_release(tmp_path, mocker):
    (tmp_path / "parquet").mkdir()
    (tmp_path / "parquet" / "train.parquet").write_bytes(b"")
    _small_catalogue(tmp_path / "stac")
    mocker.patch.object(hf_publisher, "_load_parquet_as_dataset_or_dict", return_value=[])
    mocker.patch.object(hf_publisher, "_push_dataset_with_card_retry")
    mocker.patch.object(hf_publisher, "_require_card_version")
    mocker.patch.object(hf_publisher.DatasetCard, "push_to_hub", autospec=True)
    uploaded = {}

    def read_staged_upload(folder_path, **kwargs):
        staged = Path(folder_path)
        uploaded["item"] = json.loads((staged / "project-1" / "project-1.json").read_text())
        uploaded["cog"] = (staged / "project-1" / "imagery.tif").read_bytes()
        uploaded["kwargs"] = kwargs

    api = mocker.patch.object(hf_publisher, "HfApi").return_value
    api.list_repo_refs.return_value.tags = []
    api.upload_folder.side_effect = read_staged_upload

    hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    steps = [call[0] for call in api.mock_calls if call[0] in ("upload_folder", "create_tag")]
    assert steps == ["upload_folder", "create_tag"]

    assets = "https://huggingface.co/datasets/hotosm/test/resolve/v0.1.0/stac"
    links = "https://huggingface.co/datasets/hotosm/test/raw/v0.1.0/stac"
    assert uploaded["item"]["assets"]["imagery"]["href"] == f"{assets}/project-1/imagery.tif"
    assert {link["href"] for link in uploaded["item"]["links"] if link["rel"] == "root"} == {
        f"{links}/collection.json"
    }
    assert uploaded["cog"] == b"cog"
    assert uploaded["kwargs"]["path_in_repo"] == "stac"
    assert uploaded["kwargs"]["delete_patterns"] == ["*"]
    assert (
        json.loads((tmp_path / "stac" / "project-1" / "project-1.json").read_text())["assets"][
            "imagery"
        ]["href"]
        == "./imagery.tif"
    )


def test_raw_upload_leaves_out_the_stac_folder(tmp_path, mocker):
    api = mocker.patch.object(hf_publisher, "HfApi").return_value

    hf_publisher.push_raw_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    assert "stac/*" in api.upload_folder.call_args.kwargs["ignore_patterns"]


def _dataset_ready_to_push(tmp_path, mocker, card_version, chip):
    (tmp_path / "parquet").mkdir()
    (tmp_path / "parquet" / "train.parquet").write_bytes(b"")
    stats = _upsert_generated_stats_block(
        CARD_TEMPLATE,
        _generated(tmp_path, [_row(chip)] * 8, card_version),
        split_stats=split_stats_from_records(
            {"train": [_split_row(1, "Ghana", 5)] * 8, "validation": [], "test": []}
        ),
        splits=SplitAssignment(SplitSettings(), tiles={}, excluded={}),
    )
    (tmp_path / "dataset_card.md").write_text(stats)
    mocker.patch.object(hf_publisher, "_load_parquet_as_dataset_or_dict", return_value=[])
    data_push = mocker.patch.object(hf_publisher, "_push_dataset_with_card_retry")
    mocker.patch.object(hf_publisher, "_tag_hf_revision")
    mocker.patch.object(hf_publisher, "HfApi")
    card_push = mocker.patch.object(hf_publisher.DatasetCard, "push_to_hub", autospec=True)
    return card_push, data_push


def test_push_publishes_the_card_with_its_split_table(tmp_path, mocker, chip):
    card_push, _ = _dataset_ready_to_push(tmp_path, mocker, card_version="0.1.0", chip=chip)

    hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    pushed_card = card_push.call_args.args[0]
    assert "| Splits | train 8, validation " in str(pushed_card)


@pytest.mark.parametrize("has_samples", [True, False])
def test_push_uploads_the_sample_tiles_when_the_card_has_them(tmp_path, mocker, chip, has_samples):
    _dataset_ready_to_push(tmp_path, mocker, card_version="0.1.0", chip=chip)
    if has_samples:
        (tmp_path / "samples.png").write_bytes(b"png")

    hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    upload = hf_publisher.HfApi.return_value.upload_file
    uploaded = [call.kwargs["path_in_repo"] for call in upload.call_args_list]
    assert uploaded == (["samples.png"] if has_samples else [])


def test_push_refuses_a_card_written_for_another_version(tmp_path, mocker, chip):
    card_push, data_push = _dataset_ready_to_push(tmp_path, mocker, card_version="0.1.0", chip=chip)

    with pytest.raises(ValueError, match=r"describes version 0\.1\.0, not 0\.2\.0"):
        hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.2.0")

    card_push.assert_not_called()
    data_push.assert_not_called()


def test_push_refuses_a_card_without_generated_stats(tmp_path, mocker, chip):
    card_push, _ = _dataset_ready_to_push(tmp_path, mocker, card_version="0.1.0", chip=chip)
    (tmp_path / "dataset_card.md").write_text(CARD_TEMPLATE)

    with pytest.raises(ValueError, match="has no generated stats"):
        hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    card_push.assert_not_called()


def test_card_counts_only_rows_written_to_the_split_parquets(output_dir, chip):
    project_dir = output_dir / "project_9"
    (project_dir / "chips").mkdir(parents=True)
    (project_dir / "labels").mkdir()
    (project_dir / "metadata.json").write_text(
        json.dumps(
            {
                "project_id": 9,
                "name": "Roads",
                "country": "Nepal",
                "feature_type": "building",
                "tag_values": [],
                "required_tags": [],
            }
        )
    )
    building = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0.5, 0.5]},
        "properties": {"tags": {"building": "yes"}},
    }
    for x in (100, 101, 102):
        tile_id = f"OAM-{x}-200-19"
        shutil.copyfile(chip, project_dir / "chips" / f"{tile_id}.tif")
        (project_dir / "labels" / f"{tile_id}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": [building]})
        )
    splits = {
        "settings": {},
        "tiles": {"OAM-100-200-19": "train"},
        "excluded": {"OAM-101-200-19": "buffer"},
    }
    (output_dir / "splits.json").write_text(json.dumps(splits))

    generate_parquet(output_dir, dataset_version="0.1.0")

    card_text = (output_dir / "dataset_card.md").read_text()
    assert "| Tiles | 1 |" in card_text
    assert "leaves out" not in card_text
    wiki = "[Key:building](https://wiki.openstreetmap.org/wiki/Key:building)"
    assert f"The wiki describes how each tag is mapped: {wiki}." in card_text


def test_push_refuses_a_version_already_released(tmp_path, mocker, chip):
    card_push, data_push = _dataset_ready_to_push(tmp_path, mocker, card_version="0.1.0", chip=chip)
    mocker.patch.object(hf_publisher, "release_tags", return_value={"v0.1.0"})

    with pytest.raises(ValueError, match="bump --dataset-version"):
        hf_publisher.push_to_hub(tmp_path, "hotosm/test", dataset_version="0.1.0")

    data_push.assert_not_called()
    card_push.assert_not_called()


def test_a_failed_release_tag_raises(mocker):
    api = mocker.patch.object(hf_publisher, "HfApi").return_value
    response = httpx.Response(409, request=httpx.Request("POST", "https://huggingface.co"))
    api.create_tag.side_effect = HfHubHTTPError("tag exists", response=response)

    with pytest.raises(HfHubHTTPError):
        hf_publisher._tag_hf_revision("hotosm/test", "0.1.0")  # noqa: SLF001
