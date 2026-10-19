import json
import math
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pystac
import pytest
import rasterio
from rasterio.transform import from_bounds

from hot_oam_dataset import stac_export
from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.oam_metadata import OamImagery
from hot_oam_dataset.split_generator import SPLIT_NAMES
from hot_oam_dataset.stac_export import (
    LABEL_EXTENSION_SCHEMA,
    PARQUET_COLUMNS,
    generate_stac,
    labels_frame,
    write_imagery_cog,
)

ZOOM = 19
TILE_PIXELS = 256


def _chip(path, tile, colour):
    bounds = mercantile.bounds(tile)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=TILE_PIXELS,
        height=TILE_PIXELS,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(
            bounds.west, bounds.south, bounds.east, bounds.north, TILE_PIXELS, TILE_PIXELS
        ),
    ) as chip:
        chip.write(np.full((3, TILE_PIXELS, TILE_PIXELS), colour, dtype=np.uint8))


def _record(tmp_path, x, y, split, colour, features=(), project_id=1):
    tile = mercantile.Tile(x, y, ZOOM)
    tile_id = f"OAM-{x}-{y}-{ZOOM}"
    chip_path = tmp_path / f"{tile_id}.tif"
    _chip(chip_path, tile, colour)
    bounds = mercantile.bounds(tile)
    return {
        "image": {"bytes": None, "path": str(chip_path)},
        "mask": None,
        "tile_id": tile_id,
        "tile_x": x,
        "tile_y": y,
        "tile_z": ZOOM,
        "project_id": project_id,
        "project_url": f"https://tasks.hotosm.org/projects/{project_id}",
        "project_name": "Tree mapping",
        "country": "Sierra Leone",
        "organisation": "HOT",
        "imagery_url": "https://tiles.openaerialmap.org/x/{z}/{x}/{y}",
        "feature_type": "natural",
        "num_features": len(features),
        "class_counts": "{}",
        "label_geojson": json.dumps({"type": "FeatureCollection", "features": list(features)}),
        "bbox_west": bounds.west,
        "bbox_south": bounds.south,
        "bbox_east": bounds.east,
        "bbox_north": bounds.north,
        "split": split,
    }


def _tree(lon, lat, value="tree"):
    return {
        "type": "Feature",
        "properties": {"osm_id": 7, "osm_type": "nodes", "tags": {"natural": value}},
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def test_imagery_cog_places_each_chip_at_its_tile_and_masks_missing_tiles(tmp_path):
    records = [
        _record(tmp_path, 100, 200, "train", 40),
        _record(tmp_path, 101, 200, "train", 200),
        _record(tmp_path, 100, 201, "test", 90),
    ]
    output = tmp_path / "imagery.tif"

    write_imagery_cog(records, output)

    with rasterio.open(output) as cog:
        assert cog.crs.to_epsg() == 3857
        assert cog.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
        top_left = mercantile.xy_bounds(100, 200, ZOOM)
        assert cog.bounds.left == pytest.approx(top_left.left)
        assert cog.bounds.top == pytest.approx(top_left.top)
        pixels = cog.read(1)
        mask = cog.dataset_mask()
    centre = TILE_PIXELS // 2
    assert abs(int(pixels[centre, centre]) - 40) <= 3
    assert abs(int(pixels[centre, TILE_PIXELS + centre]) - 200) <= 3
    assert abs(int(pixels[TILE_PIXELS + centre, centre]) - 90) <= 3
    assert mask[TILE_PIXELS + centre, TILE_PIXELS + centre] == 0
    assert mask[centre, centre] == 255


def test_imagery_cog_skips_the_empty_blocks_between_distant_tiles(tmp_path):
    records = [_record(tmp_path, 100, 200, "train", 40), _record(tmp_path, 200, 300, "train", 90)]
    output = tmp_path / "imagery.tif"

    write_imagery_cog(records, output)

    assert output.stat().st_size < 1_000_000
    with rasterio.open(output) as cog:
        assert cog.width == 101 * TILE_PIXELS


def test_imagery_cog_rejects_mixed_zoom_levels(tmp_path):
    records = [_record(tmp_path, 100, 200, "train", 40), _record(tmp_path, 101, 200, "train", 9)]
    records[1]["tile_z"] = ZOOM + 1

    with pytest.raises(ValueError, match="share a zoom level"):
        write_imagery_cog(records, tmp_path / "imagery.tif")


def test_labels_carry_the_tag_value_as_class_and_the_tile_split(tmp_path):
    records = [
        _record(tmp_path, 100, 200, "train", 40, [_tree(0.1, 0.1), _tree(0.2, 0.2, "tree_group")]),
        _record(tmp_path, 101, 200, "validation", 40, [_tree(0.3, 0.3)]),
    ]

    labels = labels_frame(records)

    assert labels.crs.to_epsg() == 4326
    assert list(labels["class"]) == ["tree", "tree_group", "tree"]
    assert list(labels["split"]) == ["train", "train", "validation"]
    assert json.loads(labels["tags"].iloc[1]) == {"natural": "tree_group"}


def _write_dataset(tmp_path, records):
    dataset_dir = tmp_path / "fAIr-guru-td-trees"
    (dataset_dir / "parquet").mkdir(parents=True)
    for split in ("train", "validation", "test"):
        rows = [{k: r[k] for k in PARQUET_COLUMNS} for r in records if r["split"] == split]
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), dataset_dir / "parquet" / f"{split}.parquet")
    (dataset_dir / "dataset_stats.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-10-15T00:00:00+00:00",
                "label_noun": "trees",
                "feature_type": "natural",
                "class_counts": {"tree": 2},
                "projects": [
                    {"project_id": 1, "mapping_types": ["OTHER"]},
                    {"project_id": 2, "mapping_types": ["BUILDINGS", "ROADS"]},
                ],
            }
        )
    )
    for project_id in (1, 2):
        project_dir = dataset_dir / f"project_{project_id}"
        project_dir.mkdir()
        metadata = {
            "imagery_url": "https://tiles.openaerialmap.org/x/{z}/{x}/{y}",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "feature_type": "natural",
            "tag_values": ["tree"],
            "required_tags": [],
        }
        (project_dir / "metadata.json").write_text(json.dumps(metadata))
    return dataset_dir


def test_stac_collection_has_label_and_imagery_items_per_project(tmp_path, monkeypatch, mocker):
    centre = mercantile.bounds(100, 200, ZOOM)
    tree = _tree((centre.west + centre.east) / 2, (centre.south + centre.north) / 2)
    records = [
        _record(tmp_path, 100, 200, "train", 40, [tree]),
        _record(tmp_path, 101, 200, "test", 60),
        _record(tmp_path, 300, 400, "validation", 80, [tree], project_id=2),
    ]
    dataset_dir = _write_dataset(tmp_path, records)
    monkeypatch.chdir(tmp_path)
    captured = datetime(2025, 10, 22, tzinfo=UTC)
    lookup = mocker.patch.object(
        stac_export, "oam_imagery", return_value=OamImagery("Freetown", captured)
    )
    card_text = CardText.from_mapping(
        {
            "repo_id": "hotosm/trees",
            "title": "Trees",
            "label_noun": "trees and shrubs",
            "tasks": ["detection", "classification"],
            "summary": "Tree data.",
            "selection": "Points.",
            "limitations": [],
        }
    )

    collection_path = generate_stac(Path(dataset_dir.name), "0.1.0", card_text=card_text)

    collection = pystac.Collection.from_file(str(collection_path))
    assert collection.title == "Trees"
    assert collection.description.startswith("Imagery, trees and shrubs and train")
    assert collection.summaries.get_list("label:tasks") == ["detection", "classification"]
    assert collection.summaries.get_list("dataset:zoom") == [ZOOM]
    items = {item.id: item for item in collection.get_items()}
    assert set(items) == {"project-1", "project-1-imagery", "project-2", "project-2-imagery"}

    labels_item = items["project-1"]
    assert labels_item.stac_extensions == [LABEL_EXTENSION_SCHEMA]
    assert labels_item.properties["label:tasks"] == ["detection", "classification"]
    assert labels_item.properties["label:type"] == "vector"
    assert labels_item.properties["label:properties"] == ["class"]
    assert labels_item.properties["label:methods"] == ["manual"]
    assert labels_item.properties["label:classes"] == [{"name": "class", "classes": ["tree"]}]
    assert labels_item.properties["label:overviews"] == [
        {"property_key": "class", "counts": [{"name": "tree", "count": 1}]}
    ]
    assert labels_item.properties["title"] == "Tree mapping"
    assert labels_item.properties["dataset:labels"] == 1
    assert items["project-2"].properties["tm:mapping_types"] == ["BUILDINGS", "ROADS"]
    split_tiles = [labels_item.properties[f"dataset:tiles_{s}"] for s in SPLIT_NAMES]
    assert split_tiles == [1, 0, 1]
    assert labels_item.assets["labels"].media_type == pystac.MediaType.GEOJSON
    assert labels_item.assets["labels"].roles == ["labels", "labels-vector"]
    assert labels_item.assets["labels_parquet"].roles == ["labels"]

    source = labels_item.get_single_link("source")
    assert source is not None
    assert source.extra_fields["label:assets"] == ["imagery"]
    source_href = source.get_absolute_href()
    assert source_href is not None
    imagery_item = pystac.Item.from_file(source_href)
    assert imagery_item.id == "project-1-imagery"
    xyz = imagery_item.get_single_link("xyz")
    assert xyz is not None
    assert xyz.href == records[0]["imagery_url"]
    lookup.assert_any_call("https://tiles.openaerialmap.org/x/{z}/{x}/{y}", [0.0, 0.0, 1.0, 1.0])
    assert imagery_item.datetime == captured
    assert "gsd" not in imagery_item.properties
    assert "gsd" not in collection.summaries.to_dict()
    resolution = imagery_item.assets["imagery"].extra_fields["raster:spatial_resolution"]
    tiles_south = mercantile.bounds(100, 200, ZOOM).south
    tiles_north = mercantile.bounds(101, 200, ZOOM).north
    centre_latitude = math.radians((tiles_south + tiles_north) / 2)
    assert resolution == pytest.approx(
        156543.03392 * math.cos(centre_latitude) / 2**ZOOM, rel=0.001
    )
    assert collection.extent.temporal.intervals[0] == [
        captured,
        datetime(2026, 10, 15, tzinfo=UTC),
    ]
    wiki = [link.href for link in collection.get_links("describedby")]
    assert wiki == [
        "https://wiki.openstreetmap.org/wiki/Key:natural",
        "https://wiki.openstreetmap.org/wiki/Tag:natural=tree",
    ]
    with rasterio.open(items["project-2-imagery"].assets["imagery"].get_absolute_href()) as cog:
        assert cog.width == TILE_PIXELS

    for item in items.values():
        for asset in item.assets.values():
            assert not asset.href.startswith("/")
            href = asset.get_absolute_href()
            assert href is not None
            assert Path(href).is_file()

    tiles = gpd.read_parquet(labels_item.assets["tiles"].get_absolute_href())
    assert tiles.crs.to_epsg() == 4326
    assert sorted(tiles["split"]) == ["test", "train"]
    geojson_labels = gpd.read_file(labels_item.assets["labels"].get_absolute_href())
    parquet_labels = gpd.read_parquet(labels_item.assets["labels_parquet"].get_absolute_href())
    assert list(geojson_labels["class"]) == list(parquet_labels["class"]) == ["tree"]
    assert list(parquet_labels["split"]) == ["train"]
    assert {str(link.title).split(":")[0] for link in collection.get_links("license")} == {
        "Imagery",
        "Labels",
    }
    assert (Path(collection_path).parent / "split_map.png").stat().st_size > 0
