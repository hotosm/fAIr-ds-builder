import json

import pytest

from hot_oam_dataset.split_generator import (
    SPLIT_NAMES,
    SplitSettings,
    _buffer_exclusions,
    collect_tiles,
    generate_splits,
    load_splits,
    tile_footprints,
)

EQUATOR_X, EQUATOR_Y = 262144, 262144


def _write_project(dataset_dir, project_id, tiles, features_by_tile=None):
    project_dir = dataset_dir / f"project_{project_id}"
    (project_dir / "labels").mkdir(parents=True)
    (project_dir / "metadata.json").write_text(json.dumps({"project_id": project_id}))
    for x, y in tiles:
        features = [
            {"type": "Feature", "geometry": None, "properties": {"osm_type": "ways", "osm_id": i}}
            for i in (features_by_tile or {}).get((x, y), [])
        ]
        (project_dir / "labels" / f"OAM-{x}-{y}-19.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features})
        )
    (project_dir / "labels" / "osm-result.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []})
    )


def _splits(dataset_dir, settings):
    splits = generate_splits(dataset_dir, settings)
    assert splits is not None
    return splits


def _grid(x0, y0, width, height):
    return [(x0 + dx, y0 + dy) for dx in range(width) for dy in range(height)]


def _split_of_cells(splits):
    return {
        tuple(int(part) for part in tile_id.split("-")[1:3]): name
        for tile_id, name in splits.tiles.items()
    }


def _is_connected(cells):
    cells = set(cells)
    seen, frontier = set(), [next(iter(cells))]
    while frontier:
        x, y = frontier.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        frontier.extend(n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if n in cells)
    return seen == cells


def test_collect_tiles_ignores_the_whole_area_label(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 2, 2))

    tiles = collect_tiles(tmp_path)

    assert len(tiles) == 4
    assert all(t.tile_id.startswith("OAM-") for t in tiles)


def test_default_split_keeps_every_tile_and_holds_out_contiguous_regions(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 40, 40))

    splits = _splits(tmp_path, SplitSettings())

    cells = _split_of_cells(splits)
    assert len(cells) == 1600
    assert splits.excluded == {}
    for held_out in ("validation", "test"):
        region = [cell for cell, name in cells.items() if name == held_out]
        assert 80 <= len(region) <= 240
        assert _is_connected(region)


def test_every_project_spans_all_splits(tmp_path):
    for project_id, offset in enumerate([0, 1000, 2000], start=1):
        _write_project(tmp_path, project_id, _grid(EQUATOR_X + offset, EQUATOR_Y, 20, 20))

    splits = _splits(tmp_path, SplitSettings())

    per_project = {}
    for tile in collect_tiles(tmp_path):
        per_project.setdefault(tile.project_id, set()).add(splits.split_for(tile.tile_id))
    assert all(names == set(SPLIT_NAMES) for names in per_project.values())


def test_project_smaller_than_the_region_count_goes_to_train(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 3, 3))

    splits = _splits(tmp_path, SplitSettings())

    assert set(splits.tiles.values()) == {"train"}


def test_buffer_leaves_out_tiles_near_a_higher_ranked_split(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 40, 40))

    splits = _splits(tmp_path, SplitSettings(buffer_meters=100))

    footprints = tile_footprints(collect_tiles(tmp_path)).set_index("tile_id")
    kept = footprints.loc[list(splits.tiles)]
    kept["split"] = [splits.tiles[tile_id] for tile_id in kept.index]
    for stronger, weaker in (("test", "validation"), ("test", "train"), ("validation", "train")):
        distances = kept[kept["split"] == weaker].distance(
            kept[kept["split"] == stronger].union_all()
        )
        assert distances.min() > 100
    assert set(splits.excluded.values()) == {"buffer"}
    assert len(splits.tiles) + len(splits.excluded) == 1600


def test_shared_feature_rule_keeps_each_feature_in_one_split(tmp_path):
    tiles = _grid(EQUATOR_X, EQUATOR_Y, 40, 40)
    road_across_project = {(x, EQUATOR_Y + 20): [7] for x in range(EQUATOR_X, EQUATOR_X + 40)}
    _write_project(tmp_path, 1, tiles, road_across_project)

    without_rule = _splits(tmp_path, SplitSettings())
    with_rule = _splits(tmp_path, SplitSettings(exclude_shared_features=True))

    road_tiles = {f"OAM-{x}-{y}-19" for x, y in road_across_project}
    assert len({without_rule.tiles[t] for t in road_tiles}) > 1
    assert len({with_rule.tiles[t] for t in road_tiles if t in with_rule.tiles}) == 1
    assert set(with_rule.excluded.values()) == {"shared_feature"}


def test_assignment_is_deterministic(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 24, 24))

    assert _splits(tmp_path, SplitSettings()).tiles == (_splits(tmp_path, SplitSettings()).tiles)


def test_splits_round_trip(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 24, 24))
    written = _splits(tmp_path, SplitSettings(buffer_meters=50))

    loaded = load_splits(tmp_path)

    assert loaded is not None
    assert loaded.settings == written.settings
    assert loaded.tiles == written.tiles
    assert loaded.excluded == written.excluded


def test_load_splits_rejects_the_older_block_format(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({"block_size": 8, "blocks": {}}))

    with pytest.raises(ValueError, match="rerun --generate-splits"):
        load_splits(tmp_path)


def test_load_splits_missing_file(tmp_path):
    assert load_splits(tmp_path) is None


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"block_size": 8}, "unknown split settings"),
        ({"regions_per_project": 2}, "must exceed"),
        ({"test_regions": 0}, "at least one"),
        ({"buffer_meters": -1}, "must not be negative"),
        ({"exclude_shared_features": "yes"}, "true or false"),
    ],
)
def test_invalid_split_settings_are_rejected(mapping, message):
    with pytest.raises(ValueError, match=message):
        SplitSettings.from_mapping(mapping)


def test_a_feature_shared_by_test_and_validation_stays_in_test(tmp_path):
    _write_project(tmp_path, 1, _grid(EQUATOR_X, EQUATOR_Y, 40, 40))
    plain = _split_of_cells(_splits(tmp_path, SplitSettings()))
    test_cell = next(cell for cell, name in plain.items() if name == "test")
    validation_cell = next(cell for cell, name in plain.items() if name == "validation")
    feature = {"type": "Feature", "geometry": None, "properties": {"osm_type": "ways", "osm_id": 7}}
    for x, y in (test_cell, validation_cell):
        label = tmp_path / "project_1" / "labels" / f"OAM-{x}-{y}-19.geojson"
        label.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))

    ruled = _splits(tmp_path, SplitSettings(exclude_shared_features=True))

    x, y = validation_cell
    assert ruled.excluded == {f"OAM-{x}-{y}-19": "shared_feature"}
    assert _split_of_cells(ruled)[test_cell] == "test"


def test_tiles_left_out_by_the_buffer_do_not_push_the_buffer_further(tmp_path):
    row = [(EQUATOR_X + dx, EQUATOR_Y) for dx in range(10)]
    _write_project(tmp_path, 1, row)
    tiles = collect_tiles(tmp_path)
    names = ["test"] * 3 + ["validation"] + ["train"] * 6
    splits = {tile.tile_id: name for tile, name in zip(tiles, names, strict=True)}

    excluded = _buffer_exclusions(tiles, splits, buffer_meters=100)

    # One tile is about 76 m wide at zoom 19, so only the validation tile and the first train tile
    # lie within 100 m of a test tile; the next train tile is close only to the dropped validation
    assert excluded == {tiles[3].tile_id, tiles[4].tile_id}
