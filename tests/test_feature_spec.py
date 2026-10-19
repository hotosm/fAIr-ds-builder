import json

import pytest

from hot_oam_dataset.feature_spec import (
    GEOMETRY_TYPES,
    TM_MAPPING_TYPES,
    dataset_wiki_pages,
    osm_wiki_pages,
    resolve_feature_spec,
)


def test_building_maps_to_buildings_and_polygon():
    spec = resolve_feature_spec("building")

    assert spec.tm_mapping_type == "BUILDINGS"
    assert spec.geometry_types == ("polygon",)
    assert spec.label_noun == "buildings"


def test_highway_maps_to_roads_and_line():
    spec = resolve_feature_spec("highway")

    assert spec.tm_mapping_type == "ROADS"
    assert spec.geometry_types == ("line",)
    assert spec.label_noun == "roads"


def test_unknown_tag_falls_back_to_other_mapping_type():
    spec = resolve_feature_spec("amenity")

    assert spec.osm_tag == "amenity"
    assert spec.tm_mapping_type == "OTHER"
    assert "line" in spec.geometry_types
    assert "polygon" in spec.geometry_types


def test_explicit_overrides_win_over_defaults():
    spec = resolve_feature_spec("highway", tm_mapping_type="OTHER", geometry_types=["polygon"])

    assert spec.osm_tag == "highway"
    assert spec.tm_mapping_type == "OTHER"
    assert spec.geometry_types == ("polygon",)


def test_tag_values_default_to_any_value():
    assert resolve_feature_spec("natural").raw_data_filters() == {
        "tags": {"all_geometry": {"join_or": {"natural": []}}}
    }


def test_required_tags_keep_the_feature_tag_values():
    spec = resolve_feature_spec("surface", tag_values=["dirt"], required_tags=["highway"])

    assert spec.raw_data_filters() == {
        "tags": {"all_geometry": {"join_and": {"surface": ["dirt"], "highway": []}}}
    }


@pytest.mark.parametrize("bad_key", ["", "highway'; --", "surface"])
def test_empty_quoted_or_repeated_required_tags_are_rejected(bad_key):
    with pytest.raises(ValueError, match="Required tags must be"):
        resolve_feature_spec("surface", required_tags=["highway", bad_key])


@pytest.mark.parametrize("bad_value", ["", "tree'; --"])
def test_empty_or_quoted_tag_values_are_rejected(bad_value):
    with pytest.raises(ValueError, match="non-empty and free of quotes"):
        resolve_feature_spec("natural", tag_values=["tree", bad_value])


@pytest.mark.parametrize("osm_tag", ["building", "highway", "waterway", "landuse", "amenity"])
def test_resolved_values_stay_within_api_vocabularies(osm_tag):
    """Both vocabularies are fixed by the TM and Raw Data APIs, so no tag may escape them."""
    spec = resolve_feature_spec(osm_tag)

    assert spec.tm_mapping_type in TM_MAPPING_TYPES
    assert all(geometry in GEOMETRY_TYPES for geometry in spec.geometry_types)


def test_wiki_pages_cover_the_key_its_values_and_required_keys():
    pages = osm_wiki_pages("surface", ["asphalt"], ["highway"])

    assert pages == [
        ("Key:surface", "https://wiki.openstreetmap.org/wiki/Key:surface"),
        ("Tag:surface=asphalt", "https://wiki.openstreetmap.org/wiki/Tag:surface=asphalt"),
        ("Key:highway", "https://wiki.openstreetmap.org/wiki/Key:highway"),
    ]


def test_dataset_wiki_pages_merge_the_projects_without_repeats(tmp_path):
    for project_id, values in ((1, ["tree"]), (2, ["tree", "tree_group"])):
        project_dir = tmp_path / f"project_{project_id}"
        project_dir.mkdir()
        metadata = {"feature_type": "natural", "tag_values": values, "required_tags": []}
        (project_dir / "metadata.json").write_text(json.dumps(metadata))

    pages = [page for page, _ in dataset_wiki_pages(tmp_path)]

    assert pages == ["Key:natural", "Tag:natural=tree", "Tag:natural=tree_group"]
