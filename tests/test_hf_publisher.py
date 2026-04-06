import json

import pytest

from hot_oam_dataset.hf_publisher import _upsert_generated_stats_block

CARD_TEMPLATE = """\
# Dataset Card
Some content here.

<!-- AUTO_STATS_START -->
<!-- AUTO_STATS_END -->
"""


def _write_dataset_stats(dataset_dir, projects, country_details=None):
    stats = {
        "total_projects": len(projects),
        "tiles_with_buildings": sum(p["valid_pairs"] for p in projects),
        "tiles_without_buildings": 0,
        "total_building_polygons": sum(p.get("total_building_polygons", 0) for p in projects),
        "avg_buildings_per_tile": 5.0,
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


def test_stats_block_includes_split_table(dataset_dir):
    projects = [
        {"project_id": 1, "valid_pairs": 800, "total_building_polygons": 5000, "country": "Kenya"},
        {"project_id": 2, "valid_pairs": 100, "total_building_polygons": 600, "country": "Peru"},
        {"project_id": 3, "valid_pairs": 100, "total_building_polygons": 400, "country": "Ghana"},
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
        {"project_id": 1, "valid_pairs": 100, "total_building_polygons": 50, "country": "Kenya"},
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
