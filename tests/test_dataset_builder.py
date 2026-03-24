import json
from pathlib import Path

import httpx
import pytest

from hot_oam_dataset.dataset_builder import (
    _lon_lat_to_tile,
    probe_imagery_url,
    validate_and_clean_project,
)

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
    assert stats["pairs_with_buildings"] == 1
    assert stats["pairs_without_buildings"] == 0
    assert stats["total_building_polygons"] == 3


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
    assert stats["pairs_without_buildings"] == 1
    assert stats["pairs_with_buildings"] == 0


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
    assert stats["pairs_with_buildings"] == 1
    assert stats["pairs_without_buildings"] == 1
    assert stats["total_building_polygons"] == 2
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


def _mock_stream(mocker, status_code: int):
    ctx = mocker.MagicMock()
    ctx.__enter__ = mocker.MagicMock(return_value=mocker.MagicMock(status_code=status_code))
    ctx.__exit__ = mocker.MagicMock(return_value=False)
    return mocker.patch("hot_oam_dataset.dataset_builder.httpx.stream", return_value=ctx)


def test_probe_returns_true_on_200(mocker):
    _mock_stream(mocker, 200)
    assert probe_imagery_url(OAM_URL, BBOX, 19) is True


def test_probe_returns_false_on_404(mocker):
    _mock_stream(mocker, 404)
    assert probe_imagery_url(OAM_URL, BBOX, 19) is False


def test_probe_returns_false_on_network_error(mocker):
    mocker.patch(
        "hot_oam_dataset.dataset_builder.httpx.stream",
        side_effect=httpx.RequestError("timeout"),
    )
    assert probe_imagery_url(OAM_URL, BBOX, 19) is False


def test_probe_substitutes_tile_coords(mocker):
    captured = {}

    def fake_stream(method, url, **kwargs):
        captured["url"] = url
        ctx = mocker.MagicMock()
        ctx.__enter__ = mocker.MagicMock(return_value=mocker.MagicMock(status_code=200))
        ctx.__exit__ = mocker.MagicMock(return_value=False)
        return ctx

    mocker.patch("hot_oam_dataset.dataset_builder.httpx.stream", side_effect=fake_stream)
    probe_imagery_url(OAM_URL, BBOX, 19)

    assert "{z}" not in captured["url"]
    assert "{x}" not in captured["url"]
    assert "{y}" not in captured["url"]


def test_probe_skips_on_empty_bbox(mocker):
    stream = mocker.patch("hot_oam_dataset.dataset_builder.httpx.stream")
    assert probe_imagery_url(OAM_URL, [], 19) is True
    stream.assert_not_called()
