import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from hot_oam_dataset.sample_tiles import pick_sample_tiles, write_sample_tiles


@pytest.fixture
def chip(tmp_path):
    path = tmp_path / "chip.tif"
    profile = {"driver": "GTiff", "width": 16, "height": 16, "count": 3, "dtype": "uint8"}
    pixel = 0.0007 / 16
    profile |= {"crs": "EPSG:4326", "transform": from_origin(0, 0.0007, pixel, pixel)}
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.full((3, 16, 16), 120, dtype=np.uint8))
    return str(path)


def _tile(index, classes, image="chip.tif"):
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0.0003, 0.0003]},
            "properties": {"tags": {"surface": name}},
        }
        for name, count in classes.items()
        for _ in range(count)
    ]
    return {
        "image": image,
        "tile_id": f"OAM-{index}-1-19",
        "num_features": len(features),
        "class_counts": json.dumps(classes),
        "label_geojson": json.dumps({"type": "FeatureCollection", "features": features}),
        "bbox_west": 0.0,
        "bbox_south": 0.0,
        "bbox_east": 0.0007,
        "bbox_north": 0.0007,
    }


def test_each_common_class_gets_one_of_the_ten_tiles_richest_in_it():
    tiles = [_tile(i, {"asphalt": 40}) for i in range(20)]
    tiles += [_tile(100 + count, {"gravel": count}) for count in range(1, 31)]

    picked = pick_sample_tiles(tiles, count=2, seed=0)

    assert [name for _, name in picked] == ["asphalt", "gravel"]
    assert json.loads(picked[1][0]["class_counts"])["gravel"] > 20


def test_a_class_already_shown_by_another_pick_is_skipped():
    tiles = [_tile(1, {"A": 5, "B": 5}), _tile(2, {"C": 90})]

    picked = pick_sample_tiles(tiles, count=6, seed=0)

    assert sorted(record["tile_id"] for record, _ in picked) == ["OAM-1-1-19", "OAM-2-1-19"]


def test_a_class_under_one_percent_is_left_to_the_random_fill():
    tiles = [_tile(i, {"yes": 100}) for i in range(10)] + [_tile(99, {"tent": 1})]

    picked = pick_sample_tiles(tiles, count=4, seed=0)

    assert {name for _, name in picked} == {"yes"}
    assert len(picked) == 4


def test_the_random_fill_draws_from_the_tiles_with_the_most_labels():
    sparse = [_tile(i, {"tree": 1}) for i in range(15)]
    tiles = sparse + [_tile(50 + i, {"tree": 9}) for i in range(5)]

    picked = pick_sample_tiles(tiles, count=5, seed=0)

    assert all(record["num_features"] == 9 for record, _ in picked)


def test_equal_label_counts_fall_back_to_the_largest_labels():
    tiles = []
    for index in range(20):
        tile = _tile(index, {"solar": 1})
        size = 0.00001 * (index + 1)
        panel = [[0, 0], [size, 0], [size, size], [0, size], [0, 0]]
        feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [panel]}}
        tile["label_geojson"] = json.dumps({"type": "FeatureCollection", "features": [feature]})
        tiles.append(tile)

    picked = pick_sample_tiles(tiles, count=5, seed=0)

    assert sorted(record["tile_id"] for record, _ in picked) == sorted(
        f"OAM-{index}-1-19" for index in range(15, 20)
    )


def test_sample_grid_is_written_only_when_a_tile_has_labels(tmp_path, chip):
    path = tmp_path / "samples.png"

    assert not write_sample_tiles([_tile(1, {}, chip)], "surface", 6, 0, path)
    assert not path.exists()

    assert write_sample_tiles([_tile(1, {"asphalt": 2}, chip)], "surface", 6, 0, path)
    with Image.open(path) as image:
        colours = {tuple(pixel) for pixel in np.asarray(image.convert("RGB")).reshape(-1, 3)}
    assert (120, 120, 120) in colours
    assert (255, 255, 255) in colours
