import pytest

from hot_oam_dataset.feature_spec import GEOMETRY_TYPES, TM_MAPPING_TYPES, resolve_feature_spec


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


@pytest.mark.parametrize("osm_tag", ["building", "highway", "waterway", "landuse", "amenity"])
def test_resolved_values_stay_within_api_vocabularies(osm_tag):
    """Both vocabularies are fixed by the TM and Raw Data APIs, so no tag may escape them."""
    spec = resolve_feature_spec(osm_tag)

    assert spec.tm_mapping_type in TM_MAPPING_TYPES
    assert all(geometry in GEOMETRY_TYPES for geometry in spec.geometry_types)
