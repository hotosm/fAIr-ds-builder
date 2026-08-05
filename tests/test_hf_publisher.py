import json

import pytest
from huggingface_hub import DatasetCard

from hot_oam_dataset.hf_publisher import (
    _collect_parquet_records,
    _update_dataset_card,
    _upsert_generated_stats_block,
    resolve_dataset_card,
)

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


def _write_splits(dataset_dir, assignments):
    (dataset_dir / "splits.json").write_text(json.dumps(assignments))


def _write_summary(dataset_dir, projects):
    (dataset_dir / "projects_summary.json").write_text(json.dumps(projects))


@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path


@pytest.fixture
def repo_with_template(tmp_path, monkeypatch):
    def _setup(template: str = CARD_TEMPLATE):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        template_path = repo_root / "dataset_card.md"
        template_path.write_text(template)
        monkeypatch.chdir(repo_root)

        output_dir = tmp_path / "freetown-dataset"
        output_dir.mkdir()
        return template_path, output_dir

    return _setup


def test_new_dataset_gets_own_card_and_template_is_untouched(repo_with_template):
    template_path, output_dir = repo_with_template()
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )

    _update_dataset_card(output_dir, split_counts={"train": 10}, dataset_version="0.3.0")

    card_text = (output_dir / "dataset_card.md").read_text()
    assert "Dataset version: 0.3.0" in card_text
    assert "Tile pairs (num_examples): 10" in card_text
    assert template_path.read_text() == CARD_TEMPLATE


def test_card_front_matter_splits_match_published_counts(repo_with_template):
    _, output_dir = repo_with_template(CARD_TEMPLATE_WITH_METADATA)
    _write_dataset_stats(
        output_dir,
        [{"project_id": 1, "valid_pairs": 10, "total_label_features": 5, "country": "Ghana"}],
    )

    _update_dataset_card(
        output_dir,
        split_counts={"train": 8, "validation": 1, "test": 1},
        dataset_version="0.3.0",
    )

    card = DatasetCard((output_dir / "dataset_card.md").read_text())
    assert card.data["dataset_info"]["splits"] == [
        {"name": "train", "num_examples": 8},
        {"name": "validation", "num_examples": 1},
        {"name": "test", "num_examples": 1},
    ]


def test_existing_dataset_card_is_not_reseeded(repo_with_template):
    _, output_dir = repo_with_template()
    own_card = output_dir / "dataset_card.md"
    own_card.write_text("# Freetown card\n")

    assert resolve_dataset_card(output_dir) == own_card
    assert own_card.read_text() == "# Freetown card\n"


def test_missing_template_skips_card_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    assert resolve_dataset_card(output_dir) is None


def test_stats_block_includes_split_table(dataset_dir):
    projects = [
        {"project_id": 1, "valid_pairs": 800, "total_label_features": 5000, "country": "Kenya"},
        {"project_id": 2, "valid_pairs": 100, "total_label_features": 600, "country": "Peru"},
        {"project_id": 3, "valid_pairs": 100, "total_label_features": 400, "country": "Ghana"},
    ]
    _write_dataset_stats(dataset_dir, projects)
    _write_splits(dataset_dir, {"1": "train", "2": "validation", "3": "test"})
    _write_summary(
        dataset_dir,
        [
            {"project_id": 1, "imagery_url": "https://oam.org/1"},
            {"project_id": 2, "imagery_url": "https://oam.org/2"},
            {"project_id": 3, "imagery_url": "https://oam.org/3"},
        ],
    )

    result = _upsert_generated_stats_block(CARD_TEMPLATE, dataset_dir, 1000, "0.2.0")

    assert "### Data Splits" in result
    assert "| train |" in result
    assert "| validation |" in result
    assert "| test |" in result
    assert "5,000" in result
    assert "600" in result
    assert "400" in result


def test_stats_block_omits_splits_when_no_splits_file(dataset_dir):
    projects = [
        {"project_id": 1, "valid_pairs": 100, "total_label_features": 50, "country": "Kenya"},
    ]
    _write_dataset_stats(dataset_dir, projects)
    _write_summary(
        dataset_dir,
        [{"project_id": 1, "imagery_url": "https://oam.org/1"}],
    )

    result = _upsert_generated_stats_block(CARD_TEMPLATE, dataset_dir, 100, "0.2.0")

    assert "### Data Splits" not in result
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
