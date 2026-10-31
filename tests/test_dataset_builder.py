import asyncio
import json
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pytest
import rasterio
from geomltoolkits.downloader import osm as osm_downloader
from geomltoolkits.downloader import tms as tms_downloader
from rasterio.transform import from_origin
from shapely.geometry import shape

from hot_oam_dataset.dataset_builder import (
    DatasetConfig,
    _build_aggregate_report,
    _collect_per_project_stats,
    _download_chips_and_labels,
    _lon_lat_to_tile,
    _save_project_metadata,
    _write_buffered_labels,
    build_project_dataset,
    probe_imagery_url,
    remove_import_attribute_tags,
    validate_and_clean_project,
)
from hot_oam_dataset.feature_spec import label_class_counts
from hot_oam_dataset.tm_client import TaskingManagerProject

OAM_URL = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
BBOX = [96.0, 21.9, 96.1, 22.0]
AOI_SQUARE = {
    "type": "Polygon",
    "coordinates": [[[96.0, 21.9], [96.1, 21.9], [96.1, 22.0], [96.0, 22.0], [96.0, 21.9]]],
}
VALIDATED_HALF = {
    "type": "Polygon",
    "coordinates": [[[96.0, 21.9], [96.05, 21.9], [96.05, 22.0], [96.0, 22.0], [96.0, 21.9]]],
}


def _chip(path: Path, transparent_pixels: int = 0):
    """Write a 64x64 chip; transparent pixels add an alpha band, as in PNG tiles with gaps."""
    pixels = np.full((3, 64, 64), 120, dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "width": 64,
        "height": 64,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, 64, 1, 1),
    }
    if transparent_pixels:
        alpha = np.full((1, 64, 64), 255, dtype=np.uint8)
        alpha.flat[:transparent_pixels] = 0
        pixels = np.concatenate([pixels, alpha])
        profile |= {"count": 4, "photometric": "RGB", "alpha": "YES"}
    with rasterio.open(path, "w", **profile) as chip:
        chip.write(pixels)


def _corrupt_chip(path: Path):
    path.write_bytes(b"x" * 100)


def _label(path: Path, features: int = 1):
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": {}, "properties": {}} for _ in range(features)
                ],
            }
        )
    )


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "chips").mkdir()
    (tmp_path / "labels").mkdir()
    return tmp_path


def test_valid_pair_counted(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif")
    _label(project_dir / "labels" / "OAM-1-2-19.geojson", features=3)

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 1
    assert stats["pairs_with_features"] == 1
    assert stats["pairs_without_features"] == 0
    assert stats["total_label_features"] == 3


def test_corrupt_chip_removed(project_dir):
    _corrupt_chip(project_dir / "chips" / "OAM-1-2-19.tif")
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 0
    assert stats["removed_corrupt_chips"] == 1
    assert not (project_dir / "chips" / "OAM-1-2-19.tif").exists()


@pytest.mark.parametrize(("drop_small_chips", "kept"), [(True, 0), (False, 1)])
def test_small_valid_chips_are_dropped_only_when_asked(project_dir, drop_small_chips, kept):
    chip_path = project_dir / "chips" / "OAM-1-2-23.tif"
    with rasterio.open(
        chip_path,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 256, 1, 1),
        compress="deflate",
    ) as chip:
        chip.write(np.full((3, 256, 256), 140, dtype=np.uint8))
    _label(project_dir / "labels" / "OAM-1-2-23.geojson")
    assert chip_path.stat().st_size < 5000

    stats = validate_and_clean_project(project_dir, "OAM", 1, drop_small_chips=drop_small_chips)

    assert stats["valid_pairs"] == kept
    assert chip_path.exists() == bool(kept)


def test_orphan_label_removed(project_dir):
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["removed_orphan_labels"] == 1
    assert not (project_dir / "labels" / "OAM-1-2-19.geojson").exists()


def test_empty_label_kept_as_negative_sample(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif")
    _label(project_dir / "labels" / "OAM-1-2-19.geojson", features=0)

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 1
    assert stats["pairs_without_features"] == 1
    assert stats["pairs_with_features"] == 0


def test_unpaired_chip_removed(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["removed_unpaired_chips"] == 1
    assert not (project_dir / "chips" / "OAM-1-2-19.tif").exists()


def test_multiple_pairs_mixed(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif")
    _label(project_dir / "labels" / "OAM-1-2-19.geojson", features=2)

    _chip(project_dir / "chips" / "OAM-1-3-19.tif")
    _label(project_dir / "labels" / "OAM-1-3-19.geojson", features=0)

    _corrupt_chip(project_dir / "chips" / "OAM-1-4-19.tif")
    _label(project_dir / "labels" / "OAM-1-4-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 2
    assert stats["pairs_with_features"] == 1
    assert stats["pairs_without_features"] == 1
    assert stats["total_label_features"] == 2
    assert stats["removed_corrupt_chips"] == 1


def test_chip_with_missing_imagery_is_removed_with_its_label_and_mask(project_dir):
    (project_dir / "masks").mkdir()
    _chip(project_dir / "chips" / "OAM-1-2-19.tif", transparent_pixels=64 * 64 // 50)
    _chip(project_dir / "masks" / "OAM-1-2-19.tif")
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")
    _chip(project_dir / "chips" / "OAM-1-3-19.tif")
    _label(project_dir / "labels" / "OAM-1-3-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["removed_missing_imagery_chips"] == 1
    assert stats["removed_orphan_labels"] == 0
    assert stats["valid_pairs"] == 1
    assert not (project_dir / "chips" / "OAM-1-2-19.tif").exists()
    assert not (project_dir / "masks" / "OAM-1-2-19.tif").exists()
    assert not (project_dir / "labels" / "OAM-1-2-19.geojson").exists()


def test_chip_with_a_few_transparent_pixels_is_kept(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif", transparent_pixels=40)
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["removed_missing_imagery_chips"] == 0
    assert stats["valid_pairs"] == 1


def test_dark_pixels_without_transparency_are_kept(project_dir):
    chip_path = project_dir / "chips" / "OAM-1-2-19.tif"
    _chip(chip_path)
    with rasterio.open(chip_path, "r+") as chip:
        chip.write(np.zeros((3, 64, 64), dtype=np.uint8))
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["removed_missing_imagery_chips"] == 0
    assert stats["valid_pairs"] == 1


def test_lon_lat_to_tile_known():
    x, y = _lon_lat_to_tile(96.05, 21.95, 19)
    assert isinstance(x, int) and x > 0
    assert isinstance(y, int) and y > 0


def test_lon_lat_to_tile_zoom_doubles_coords():
    x18, y18 = _lon_lat_to_tile(96.05, 21.95, 18)
    x19, y19 = _lon_lat_to_tile(96.05, 21.95, 19)
    assert x19 in (x18 * 2, x18 * 2 + 1)
    assert y19 in (y18 * 2, y18 * 2 + 1)


def _mock_get(mocker, status_code: int):
    return mocker.patch(
        "hot_oam_dataset.dataset_builder.httpx.get",
        return_value=httpx.Response(status_code),
    )


def test_probe_returns_true_on_200(mocker):
    _mock_get(mocker, 200)
    assert probe_imagery_url(OAM_URL, BBOX, 19) is True


def test_probe_returns_false_on_404(mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.time.sleep")
    _mock_get(mocker, 404)
    assert probe_imagery_url(OAM_URL, BBOX, 19) is False


def test_probe_returns_false_on_network_error(mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.time.sleep")
    mocker.patch(
        "hot_oam_dataset.dataset_builder.httpx.get",
        side_effect=httpx.RequestError("timeout"),
    )
    assert probe_imagery_url(OAM_URL, BBOX, 19) is False


def test_probe_retries_on_transient_failure(mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.time.sleep")
    mocker.patch(
        "hot_oam_dataset.dataset_builder.httpx.get",
        side_effect=[
            httpx.RequestError("timeout"),
            httpx.Response(200),
        ],
    )
    assert probe_imagery_url(OAM_URL, BBOX, 19, max_retries=2) is True


def test_probe_substitutes_tile_coords(mocker):
    mock = _mock_get(mocker, 200)
    probe_imagery_url(OAM_URL, BBOX, 19)

    called_url = mock.call_args[0][0]
    assert "{z}" not in called_url
    assert "{x}" not in called_url
    assert "{y}" not in called_url


def test_probe_skips_on_empty_bbox(mocker):
    mock = mocker.patch("hot_oam_dataset.dataset_builder.httpx.get")
    assert probe_imagery_url(OAM_URL, [], 19) is True
    mock.assert_not_called()


def _project_record(project_id: int = 1) -> TaskingManagerProject:
    return TaskingManagerProject(
        project_id=project_id,
        name=f"Project {project_id}",
        status="PUBLISHED",
        percent_mapped=100,
        percent_validated=100,
        imagery_url=OAM_URL,
        area_of_interest=AOI_SQUARE,
        validated_area=VALIDATED_HALF,
        validated_task_count=1,
        total_task_count=2,
        bbox=BBOX,
        mapping_types=["ROADS"],
        organisation="HOT",
        created="2026-01-01T00:00:00Z",
        country="Nepal",
    )


def _run_download(project_dir: Path, config: DatasetConfig, mocker) -> dict:
    (project_dir / "chips").mkdir(exist_ok=True)
    (project_dir / "tiles.geojson").write_text('{"type": "FeatureCollection", "features": []}')

    captured: dict = {}

    async def fake_download_tiles(**_kwargs):
        return None

    async def fake_download_osm(**kwargs):
        captured.update(kwargs)

    mocker.patch.object(tms_downloader, "download_tiles", side_effect=fake_download_tiles)
    mocker.patch.object(
        osm_downloader, "download_osm_data", autospec=True, side_effect=fake_download_osm
    )
    mocker.patch("hot_oam_dataset.dataset_builder._flatten_label_output")
    mocker.patch("hot_oam_dataset.dataset_builder._burn_project_binary_masks")
    mocker.patch("hot_oam_dataset.dataset_builder._update_metadata_with_stats")

    asyncio.run(
        _download_chips_and_labels(
            _project_record(), project_dir, project_dir / "aoi.geojson", config
        )
    )
    return captured


def test_highway_requests_line_geometry_from_osm(project_dir, mocker):
    config = DatasetConfig(output_dir=project_dir, feature_type="highway")

    captured = _run_download(project_dir, config, mocker)

    assert captured["filters"] == {"tags": {"all_geometry": {"join_or": {"highway": []}}}}
    assert captured["geometry_types"] == ["line"]


def test_building_requests_polygon_geometry_from_osm(project_dir, mocker):
    config = DatasetConfig(output_dir=project_dir, feature_type="building")

    captured = _run_download(project_dir, config, mocker)

    assert captured["filters"] == {"tags": {"all_geometry": {"join_or": {"building": []}}}}
    assert captured["geometry_types"] == ["polygon"]


def test_osm_request_keeps_features_crossing_the_tiled_area_edge(project_dir, mocker):
    config = DatasetConfig(output_dir=project_dir, feature_type="building")

    captured = _run_download(project_dir, config, mocker)

    assert captured["use_st_within"] is False


def test_explicit_geometry_types_override_the_feature_default(project_dir, mocker):
    config = DatasetConfig(
        output_dir=project_dir, feature_type="highway", geometry_types=["polygon", "line"]
    )

    captured = _run_download(project_dir, config, mocker)

    assert captured["geometry_types"] == ["polygon", "line"]


def test_tag_values_narrow_the_osm_request_to_those_values(project_dir, mocker):
    config = DatasetConfig(
        output_dir=project_dir, feature_type="natural", tag_values=["tree", "tree_group"]
    )

    captured = _run_download(project_dir, config, mocker)

    assert captured["filters"] == {
        "tags": {"all_geometry": {"join_or": {"natural": ["tree", "tree_group"]}}}
    }


def test_metadata_records_what_produced_the_labels(tmp_path):
    project_dir = tmp_path / "project_1"
    project_dir.mkdir()
    config = DatasetConfig(
        output_dir=tmp_path,
        feature_type="highway",
        label_buffer_meters=4.0,
        tag_values=["residential"],
        required_tags=["surface"],
    )

    _save_project_metadata(_project_record(), project_dir, config)

    metadata = json.loads((project_dir / "metadata.json").read_text())
    assert metadata["feature_type"] == "highway"
    assert metadata["tm_mapping_type"] == "ROADS"
    assert metadata["geometry_types"] == ["line"]
    assert metadata["tag_values"] == ["residential"]
    assert metadata["required_tags"] == ["surface"]
    assert metadata["zoom"] == config.zoom
    assert metadata["tile_prefix"] == config.prefix
    assert metadata["label_buffer_meters"] == 4.0
    assert metadata["validated_task_count"] == 1
    assert metadata["total_task_count"] == 2


def test_tiles_are_requested_only_inside_validated_tasks(tmp_path, mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.probe_imagery_url", return_value=True)
    download = mocker.patch(
        "hot_oam_dataset.dataset_builder._download_chips_and_labels", new_callable=mocker.MagicMock
    )
    mocker.patch("hot_oam_dataset.dataset_builder.asyncio.run")

    build_project_dataset(_project_record(), DatasetConfig(output_dir=tmp_path))

    aoi_path = download.call_args.args[2]
    aoi = json.loads(aoi_path.read_text())
    assert aoi["features"][0]["geometry"] == VALIDATED_HALF


def _feature_collection(*geometries: dict, properties: dict | None = None) -> str:
    features = [
        {"type": "Feature", "properties": properties or {}, "geometry": geometry}
        for geometry in geometries
    ]
    return json.dumps({"type": "FeatureCollection", "features": features})


def test_label_files_are_cut_per_tile_without_calling_the_raw_data_api(tmp_path, mocker):
    project_dir = tmp_path / "project_1"
    project_dir.mkdir()
    label_files = tmp_path / "label_files"
    label_files.mkdir()
    tile = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    seagrass = {
        "type": "Polygon",
        "coordinates": [[[0.5, 0.2], [1.5, 0.2], [1.5, 0.8], [0.5, 0.8], [0.5, 0.2]]],
    }
    tags = {"osm_id": 7, "osm_type": "ways_poly", "tags": {"seamark:type": "seagrass"}}
    (label_files / "project_1_labels.geojson").write_text(
        _feature_collection(seagrass, properties=tags)
    )

    async def fake_download_tiles(**_kwargs):
        (project_dir / "tiles.geojson").write_text(
            _feature_collection(tile, properties={"title": "OAM-10-20-23"})
        )

    mocker.patch.object(tms_downloader, "download_tiles", side_effect=fake_download_tiles)
    raw_data_api = mocker.patch.object(osm_downloader, "download_osm_data", autospec=True)
    mocker.patch("hot_oam_dataset.dataset_builder.validate_and_clean_project", return_value={})
    mocker.patch("hot_oam_dataset.dataset_builder._update_metadata_with_stats")
    config = DatasetConfig(
        output_dir=tmp_path, burn_binary_masks=False, label_files=str(label_files)
    )

    asyncio.run(
        _download_chips_and_labels(
            _project_record(), project_dir, project_dir / "aoi.geojson", config
        )
    )

    raw_data_api.assert_not_called()
    labels = gpd.read_file(project_dir / "labels" / "OAM-10-20-23.geojson")
    assert labels["osm_id"].tolist() == [7]
    assert labels.total_bounds.tolist() == [0.5, 0.2, 1.0, 0.8]


def test_label_files_keep_tiles_inside_both_validated_tasks_and_their_coverage(tmp_path, mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.probe_imagery_url", return_value=True)
    download = mocker.patch(
        "hot_oam_dataset.dataset_builder._download_chips_and_labels", new_callable=mocker.MagicMock
    )
    mocker.patch("hot_oam_dataset.dataset_builder.asyncio.run")
    label_files = tmp_path / "label_files"
    label_files.mkdir()
    coverage = {
        "type": "Polygon",
        "coordinates": [
            [[96.025, 21.9], [96.1, 21.9], [96.1, 22.0], [96.025, 22.0], [96.025, 21.9]]
        ],
    }
    (label_files / "project_1_coverage.geojson").write_text(_feature_collection(coverage))

    build_project_dataset(
        _project_record(), DatasetConfig(output_dir=tmp_path, label_files=str(label_files))
    )

    aoi = json.loads(download.call_args.args[2].read_text())
    assert shape(aoi["features"][0]["geometry"]).bounds == (96.025, 21.9, 96.05, 22.0)


def test_label_coverage_outside_the_validated_tasks_is_an_error(tmp_path, mocker):
    mocker.patch("hot_oam_dataset.dataset_builder.probe_imagery_url", return_value=True)
    label_files = tmp_path / "label_files"
    label_files.mkdir()
    elsewhere = {
        "type": "Polygon",
        "coordinates": [[[10.0, 10.0], [10.1, 10.0], [10.1, 10.1], [10.0, 10.1], [10.0, 10.0]]],
    }
    (label_files / "project_1_coverage.geojson").write_text(_feature_collection(elsewhere))

    with pytest.raises(ValueError, match="does not overlap"):
        build_project_dataset(
            _project_record(), DatasetConfig(output_dir=tmp_path, label_files=str(label_files))
        )


def test_aggregate_lists_the_tag_values_used_across_projects():
    def collected(project_id: int, tag_values: list[str]) -> tuple[dict, dict]:
        metadata = {
            "project_id": project_id,
            "name": "p",
            "bbox": BBOX,
            "tag_values": tag_values,
            "mapping_types": ["OTHER", f"TYPE{project_id}"],
        }
        stats = {
            "valid_pairs": 1,
            "pairs_with_features": 1,
            "pairs_without_features": 0,
            "total_label_features": 1,
        }
        return metadata, stats

    report = _build_aggregate_report(
        [collected(1, ["tree_group", "tree"]), collected(2, ["tree"])], "0.3.0"
    )

    assert report["tag_values"] == ["tree", "tree_group"]
    assert [p["mapping_types"] for p in report["projects"]] == [
        ["OTHER", "TYPE1"],
        ["OTHER", "TYPE2"],
    ]


def test_project_built_without_tag_values_fails_with_a_rebuild_hint(tmp_path):
    project_dir = tmp_path / "project_7"
    project_dir.mkdir()
    metadata = {"project_id": 7, "dataset_stats": {"valid_pairs": 1}}
    (project_dir / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="rebuild project 7"):
        _collect_per_project_stats(tmp_path)


def test_buffering_turns_lines_into_areas(tmp_path):
    labels_path = tmp_path / "osm-result.geojson"
    labels_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"highway": "residential"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[85.30, 27.70], [85.31, 27.70]],
                        },
                    }
                ],
            }
        )
    )

    buffered_path = _write_buffered_labels(labels_path, buffer_meters=5.0)
    buffered = gpd.read_file(buffered_path)

    assert buffered_path != labels_path
    assert buffered.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
    assert buffered.to_crs(buffered.estimate_utm_crs()).geometry.area.iloc[0] > 0


def test_buffering_is_skipped_for_empty_labels(tmp_path):
    labels_path = tmp_path / "osm-result.geojson"
    labels_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    assert _write_buffered_labels(labels_path, buffer_meters=5.0) == labels_path


def _tagged(feature_type: str, *values):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"tags": {feature_type: v}}, "geometry": None}
            for v in values
        ],
    }


def test_class_counts_group_by_tag_value():
    counts = label_class_counts(_tagged("highway", "residential", "path", "residential"), "highway")

    assert dict(counts) == {"residential": 2, "path": 1}


def test_class_counts_ignore_features_without_the_tag():
    data = _tagged("highway", "residential")
    data["features"].append({"type": "Feature", "properties": {}, "geometry": None})
    data["features"].append({"type": "Feature", "properties": {"tags": {}}, "geometry": None})

    assert dict(label_class_counts(data, "highway")) == {"residential": 1}


def test_class_counts_follow_the_active_feature_type():
    data = _tagged("building", "house", "yes")

    assert dict(label_class_counts(data, "building")) == {"house": 1, "yes": 1}
    assert dict(label_class_counts(data, "highway")) == {}


def test_validate_records_class_counts(project_dir):
    _chip(project_dir / "chips" / "OAM-1-2-19.tif")
    (project_dir / "labels" / "OAM-1-2-19.geojson").write_text(
        json.dumps(_tagged("highway", "track", "track", "footway"))
    )

    stats = validate_and_clean_project(project_dir, "OAM", 1, "highway")

    assert stats["class_counts"] == {"track": 2, "footway": 1}
    assert stats["total_label_features"] == 3


def test_import_attributes_are_removed_from_label_tags_and_osm_tags_kept(project_dir):
    tags = {
        "building": "yes",
        "roof:material": "metal",
        "osm_uid": "9531596",
        "osm_user": "mapper",
        "osm_version": "1",
        "osm_changeset": "185067878",
        "osm_timestamp": "2026-07-03T23:53:40Z",
    }
    feature = {"type": "Feature", "geometry": None, "properties": {"version": 1, "tags": tags}}
    for name in ("osm-result.geojson", "OAM-1-2-19.geojson"):
        (project_dir / "labels" / name).write_text(
            json.dumps({"type": "FeatureCollection", "features": [feature]})
        )

    remove_import_attribute_tags(project_dir / "labels")

    for name in ("osm-result.geojson", "OAM-1-2-19.geojson"):
        properties = json.loads((project_dir / "labels" / name).read_text())["features"][0]
        assert properties["properties"] == {
            "version": 1,
            "tags": {"building": "yes", "roof:material": "metal"},
        }
