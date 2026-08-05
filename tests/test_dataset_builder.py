import asyncio
import json
from pathlib import Path

import geopandas as gpd
import httpx
import pytest
from geomltoolkits.downloader import osm as osm_downloader
from geomltoolkits.downloader import tms as tms_downloader

from hot_oam_dataset.dataset_builder import (
    DatasetConfig,
    _download_chips_and_labels,
    _lon_lat_to_tile,
    _save_project_metadata,
    _write_buffered_labels,
    probe_imagery_url,
    validate_and_clean_project,
)
from hot_oam_dataset.tm_client import TaskingManagerProject

OAM_URL = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
BBOX = [96.0, 21.9, 96.1, 22.0]


def _chip(path: Path, size: int = 10000):
    path.write_bytes(b"x" * size)


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
    _chip(project_dir / "chips" / "OAM-1-2-19.tif", size=100)
    _label(project_dir / "labels" / "OAM-1-2-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 0
    assert stats["removed_corrupt_chips"] == 1
    assert not (project_dir / "chips" / "OAM-1-2-19.tif").exists()


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

    _chip(project_dir / "chips" / "OAM-1-4-19.tif", size=100)
    _label(project_dir / "labels" / "OAM-1-4-19.geojson")

    stats = validate_and_clean_project(project_dir, "OAM", 1)

    assert stats["valid_pairs"] == 2
    assert stats["pairs_with_features"] == 1
    assert stats["pairs_without_features"] == 1
    assert stats["total_label_features"] == 2
    assert stats["removed_corrupt_chips"] == 1


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
        area_of_interest={"type": "Polygon", "coordinates": []},
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
    mocker.patch.object(osm_downloader, "download_osm_data", side_effect=fake_download_osm)
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

    assert captured["feature_type"] == "highway"
    assert captured["geometry_types"] == ["line"]


def test_building_requests_polygon_geometry_from_osm(project_dir, mocker):
    config = DatasetConfig(output_dir=project_dir, feature_type="building")

    captured = _run_download(project_dir, config, mocker)

    assert captured["feature_type"] == "building"
    assert captured["geometry_types"] == ["polygon"]


def test_explicit_geometry_types_override_the_feature_default(project_dir, mocker):
    config = DatasetConfig(
        output_dir=project_dir, feature_type="highway", geometry_types=["polygon", "line"]
    )

    captured = _run_download(project_dir, config, mocker)

    assert captured["geometry_types"] == ["polygon", "line"]


def test_metadata_records_what_produced_the_labels(tmp_path):
    project_dir = tmp_path / "project_1"
    project_dir.mkdir()
    config = DatasetConfig(output_dir=tmp_path, feature_type="highway", label_buffer_meters=4.0)

    _save_project_metadata(_project_record(), project_dir, config)

    metadata = json.loads((project_dir / "metadata.json").read_text())
    assert metadata["feature_type"] == "highway"
    assert metadata["tm_mapping_type"] == "ROADS"
    assert metadata["geometry_types"] == ["line"]
    assert metadata["zoom"] == config.zoom
    assert metadata["tile_prefix"] == config.prefix
    assert metadata["label_buffer_meters"] == 4.0


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
