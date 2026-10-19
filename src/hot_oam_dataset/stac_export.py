"""Static STAC catalogue of the split Parquets, for viewing imagery and labels in GIS tools."""

import json
import logging
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pyarrow.parquet as pq
import pystac
import rasterio
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio import (
    shutil as raster_shutil,  # ty: ignore[unresolved-import]  compiled module, no stub
)
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, shape

from hot_oam_dataset.dataset_card import CardText, metres_per_pixel
from hot_oam_dataset.feature_spec import dataset_wiki_pages
from hot_oam_dataset.oam_metadata import OamImagery, oam_imagery
from hot_oam_dataset.split_generator import SPLIT_NAMES

logger = logging.getLogger(__name__)

STAC_DIRNAME = "stac"
SPLIT_MAP_FILENAME = "split_map.png"
LABEL_CLASS_PROPERTY = "class"
# The STAC label extension fAIr dataset items use
LABEL_EXTENSION_SCHEMA = "https://stac-extensions.github.io/label/v1.0.1/schema.json"
WEB_MAP_LINKS_SCHEMA = "https://stac-extensions.github.io/web-map-links/v1.3.0/schema.json"
RASTER_EXTENSION_SCHEMA = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"
GEOPARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
SPLIT_COLORS = {"train": "#1b9e77", "validation": "#d95f02", "test": "#7570b3"}
PARQUET_COLUMNS = [
    "image",
    "mask",
    "tile_id",
    "tile_x",
    "tile_y",
    "tile_z",
    "project_id",
    "project_url",
    "project_name",
    "country",
    "organisation",
    "imagery_url",
    "feature_type",
    "num_features",
    "class_counts",
    "label_geojson",
]
# STAC Browser needs a single SPDX id; ODbL is the stricter term, the links name each part's license
LICENSE_LINKS = [
    pystac.Link(
        pystac.RelType.LICENSE,
        "https://creativecommons.org/licenses/by/4.0/",
        media_type=pystac.MediaType.HTML,
        title="Imagery: CC-BY-4.0, contributors of Open Imagery Network",
    ),
    pystac.Link(
        pystac.RelType.LICENSE,
        "https://opendatacommons.org/licenses/odbl/1-0/",
        media_type=pystac.MediaType.HTML,
        title="Labels: ODbL-1.0, OpenStreetMap contributors",
    ),
]
PROVIDERS = [
    pystac.Provider(
        "OpenAerialMap contributors",
        roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.LICENSOR],
        url="https://openaerialmap.org/",
    ),
    pystac.Provider(
        "OpenStreetMap contributors",
        roles=[pystac.ProviderRole.PRODUCER, pystac.ProviderRole.LICENSOR],
        url="https://www.openstreetmap.org/copyright",
    ),
    pystac.Provider(
        "Humanitarian OpenStreetMap Team",
        roles=[pystac.ProviderRole.PROCESSOR, pystac.ProviderRole.HOST],
        url="https://tasks.hotosm.org/",
    ),
]


def read_split_records(parquet_dir: Path) -> list[dict]:
    """Rows of every split Parquet, tagged with their split; chips are referenced by local path."""
    records = []
    for split_name in SPLIT_NAMES:
        parquet_path = parquet_dir / f"{split_name}.parquet"
        if not parquet_path.exists():
            continue
        rows = pq.read_table(parquet_path, columns=PARQUET_COLUMNS).to_pylist()
        records.extend({**row, "split": split_name} for row in rows)
    if not records:
        msg = f"No split Parquet in {parquet_dir}; run --generate-splits and --generate-parquet"
        raise FileNotFoundError(msg)
    return records


def tiles_frame(records: list[dict]) -> gpd.GeoDataFrame:
    rows = [
        {
            "tile_id": r["tile_id"],
            "split": r["split"],
            "project_id": r["project_id"],
            "num_features": r["num_features"],
            "class_counts": r["class_counts"],
            "geometry": box(*mercantile.bounds(r["tile_x"], r["tile_y"], r["tile_z"])),
        }
        for r in records
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def labels_frame(records: list[dict]) -> gpd.GeoDataFrame:
    """One row per label feature as clipped to its tile, with the tag value as the class."""
    rows = []
    for record in records:
        for feature in json.loads(record["label_geojson"])["features"]:
            properties = feature["properties"]
            rows.append(
                {
                    "tile_id": record["tile_id"],
                    "split": record["split"],
                    "project_id": record["project_id"],
                    LABEL_CLASS_PROPERTY: properties["tags"].get(record["feature_type"]),
                    "osm_id": properties.get("osm_id"),
                    "osm_type": properties.get("osm_type"),
                    "osm_timestamp": properties.get("timestamp"),
                    "tags": json.dumps(properties["tags"]),
                    "geometry": shape(feature["geometry"]),
                }
            )
    columns = ["tile_id", "split", "project_id", LABEL_CLASS_PROPERTY, "osm_id", "osm_type"]
    columns += ["osm_timestamp", "tags", "geometry"]
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")


def write_imagery_cog(records: list[dict], output_path: Path) -> None:
    """Place each chip at its XYZ grid position in Web Mercator, so no pixel is resampled."""
    zooms = {r["tile_z"] for r in records}
    if len(zooms) != 1:
        raise ValueError(f"Chips of one project must share a zoom level, found {sorted(zooms)}")
    zoom = zooms.pop()

    with rasterio.open(records[0]["image"]["path"]) as first_chip:
        tile_pixels = first_chip.width

    min_x = min(r["tile_x"] for r in records)
    min_y = min(r["tile_y"] for r in records)
    width = (max(r["tile_x"] for r in records) - min_x + 1) * tile_pixels
    height = (max(r["tile_y"] for r in records) - min_y + 1) * tile_pixels
    origin = mercantile.xy_bounds(min_x, min_y, zoom)
    pixel_size = (origin.right - origin.left) / tile_pixels

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 4,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(origin.left, origin.top, pixel_size, pixel_size),
        "photometric": "RGB",
        "alpha": "YES",
        "tiled": True,
        "blockxsize": tile_pixels,
        "blockysize": tile_pixels,
        "sparse_ok": True,
        "BIGTIFF": "IF_SAFER",
    }
    # The uncompressed mosaic can outgrow a memory-backed /tmp, so it is staged beside the output
    with tempfile.TemporaryDirectory(dir=output_path.parent) as staging_dir:
        staging_path = Path(staging_dir) / "mosaic.tif"
        with rasterio.open(staging_path, "w", **profile) as mosaic:
            for record in records:
                with rasterio.open(record["image"]["path"]) as chip:
                    pixels = np.concatenate([chip.read([1, 2, 3]), chip.dataset_mask()[np.newaxis]])
                row = (record["tile_y"] - min_y) * tile_pixels
                column = (record["tile_x"] - min_x) * tile_pixels
                window = Window.from_slices(
                    (row, row + tile_pixels), (column, column + tile_pixels)
                )
                mosaic.write(pixels, window=window)

        # SPARSE_OK skips blocks with no chip, which dominate when a project's tiles are spread out
        raster_shutil.copy(
            staging_path,
            output_path,
            driver="COG",
            COMPRESS="JPEG",
            QUALITY=90,
            OVERVIEW_RESAMPLING="AVERAGE",
            BIGTIFF="IF_SAFER",
            SPARSE_OK="TRUE",
        )


def write_split_map(tiles: gpd.GeoDataFrame, project_names: dict[int, str], path: Path) -> None:
    project_ids = sorted(tiles["project_id"].unique())
    columns = min(4, len(project_ids))
    rows = -(-len(project_ids) // columns)
    figure = Figure(figsize=(4 * columns, 4 * rows + 0.6), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).flatten()

    for axis, project_id in zip(axes, project_ids, strict=False):
        project_tiles = tiles[tiles["project_id"] == project_id].to_crs("EPSG:3857")
        for split_name in SPLIT_NAMES:
            split_tiles = project_tiles[project_tiles["split"] == split_name]
            if not split_tiles.empty:
                split_tiles.plot(ax=axis, color=SPLIT_COLORS[split_name], linewidth=0)
        axis.set_title(f"{project_id}: {project_names[project_id][:40]}", fontsize=9)
        axis.set_axis_off()
    for axis in axes[len(project_ids) :]:
        axis.set_axis_off()

    figure.legend(
        handles=[Patch(color=SPLIT_COLORS[name], label=name) for name in SPLIT_NAMES],
        loc="outside lower center",
        ncols=len(SPLIT_NAMES),
    )
    figure.savefig(path, dpi=110)


def _imagery_item(
    record: dict, tiles: gpd.GeoDataFrame, source: OamImagery, href: Path
) -> pystac.Item:
    project_id = record["project_id"]
    centre_latitude = (tiles.total_bounds[1] + tiles.total_bounds[3]) / 2
    item = pystac.Item(
        id=f"project-{project_id}-imagery",
        geometry=box(*tiles.total_bounds).__geo_interface__,
        bbox=list(tiles.total_bounds),
        datetime=source.acquired,
        properties={
            "title": f"{record['project_name']} imagery",
            "description": (
                f"OpenAerialMap imagery '{source.title}' of Tasking Manager project {project_id}: "
                "the dataset chips placed at their tile positions."
            ),
        },
        stac_extensions=[WEB_MAP_LINKS_SCHEMA, RASTER_EXTENSION_SCHEMA],
    )
    # No media type: the tile server answers in JPEG or PNG depending on the imagery
    item.add_link(pystac.Link("xyz", record["imagery_url"], title="OpenAerialMap"))
    item.add_asset(
        "imagery",
        pystac.Asset(
            href=str(href),
            title="Imagery mosaic of the dataset chips (Cloud Optimized GeoTIFF, JPEG)",
            media_type=pystac.MediaType.COG,
            roles=["visual", "data"],
            extra_fields={
                "raster:spatial_resolution": round(
                    metres_per_pixel(centre_latitude, record["tile_z"]), 4
                )
            },
        ),
    )
    return item


def _label_item(
    record: dict,
    tiles: gpd.GeoDataFrame,
    labels: gpd.GeoDataFrame,
    extracted_at: datetime,
    hrefs: dict[str, Path],
    tasks: list[str],
    mapping_types: list[str],
) -> pystac.Item:
    class_counts = Counter(value for value in labels[LABEL_CLASS_PROPERTY] if value)
    description = (
        f"OpenStreetMap `{record['feature_type']}` labels from Tasking Manager "
        f"project {record['project_id']}."
    )
    item = pystac.Item(
        id=f"project-{record['project_id']}",
        geometry=box(*tiles.total_bounds).__geo_interface__,
        bbox=list(tiles.total_bounds),
        datetime=extracted_at,
        properties={
            "title": record["project_name"],
            "description": description,
            "tm:project_id": record["project_id"],
            "tm:organisation": record["organisation"],
            "tm:mapping_types": mapping_types,
            "tm:country": record["country"],
            "dataset:zoom": record["tile_z"],
            **{
                f"dataset:tiles_{name}": int((tiles["split"] == name).sum()) for name in SPLIT_NAMES
            },
            "dataset:labels": len(labels),
            "label:description": description,
            "label:type": "vector",
            "label:properties": [LABEL_CLASS_PROPERTY],
            "label:classes": [{"name": LABEL_CLASS_PROPERTY, "classes": sorted(class_counts)}],
            "label:tasks": tasks,
            "label:methods": ["manual"],
            "label:overviews": [
                {
                    "property_key": LABEL_CLASS_PROPERTY,
                    "counts": [
                        {"name": name, "count": count} for name, count in class_counts.most_common()
                    ],
                }
            ],
        },
        stac_extensions=[LABEL_EXTENSION_SCHEMA],
    )
    item.add_link(
        pystac.Link(
            pystac.RelType.VIA,
            record["project_url"],
            media_type=pystac.MediaType.HTML,
            title="Tasking Manager project",
        )
    )
    assets = {
        "labels": pystac.Asset(
            href=str(hrefs["labels"]),
            title="Labels with OSM tags and split (GeoJSON)",
            media_type=pystac.MediaType.GEOJSON,
            roles=["labels", "labels-vector"],
        ),
        # Same labels in GeoParquet; STAC Browser draws labels only when two assets carry the role
        "labels_parquet": pystac.Asset(
            href=str(hrefs["labels_parquet"]),
            title="Labels with OSM tags and split (GeoParquet)",
            media_type=GEOPARQUET_MEDIA_TYPE,
            roles=["labels"],
        ),
        "tiles": pystac.Asset(
            href=str(hrefs["tiles"]),
            title="Tile footprints with split (GeoParquet)",
            media_type=GEOPARQUET_MEDIA_TYPE,
            roles=["metadata"],
        ),
    }
    for key, asset in assets.items():
        item.add_asset(key, asset)
    return item


def generate_stac(
    dataset_dir: Path, dataset_version: str, card_text: CardText | None = None
) -> Path:
    """Write per-project imagery and label items, with tile footprints, under one Collection."""
    records = read_split_records(dataset_dir / "parquet")
    stats = json.loads((dataset_dir / "dataset_stats.json").read_text())
    extracted_at = datetime.fromisoformat(stats["generated_at"])
    has_masks = any(r["mask"] and r["mask"]["path"] for r in records)
    tasks = list(card_text.tasks) if card_text else ["segmentation" if has_masks else "detection"]
    stac_dir = (dataset_dir / STAC_DIRNAME).resolve()
    if stac_dir.exists():
        shutil.rmtree(stac_dir)
    stac_dir.mkdir()

    tiles = tiles_frame(records)
    project_names = {r["project_id"]: r["project_name"] for r in records}
    mapping_types = {p["project_id"]: p["mapping_types"] for p in stats["projects"]}
    write_split_map(tiles, project_names, stac_dir / SPLIT_MAP_FILENAME)

    noun = card_text.label_noun if card_text else stats["label_noun"]
    collection = pystac.Collection(
        id=dataset_dir.name,
        title=card_text.title if card_text else dataset_dir.name,
        description=(
            f"Imagery, {noun} and train, validation and test tiles of {dataset_dir.name} "
            f"v{dataset_version}. Each Tasking Manager project has a labels item and an imagery "
            "item. A labels item's datetime is the label extraction time, an imagery item's the "
            "imagery capture date."
        ),
        extent=pystac.Extent(
            pystac.SpatialExtent([list(tiles.total_bounds)]),
            pystac.TemporalExtent([[extracted_at, extracted_at]]),
        ),
        license="ODbL-1.0",
        providers=PROVIDERS,
        keywords=[stats["feature_type"], "openstreetmap", "openaerialmap", "training-data"],
        stac_extensions=[LABEL_EXTENSION_SCHEMA],
    )
    collection.add_links(LICENSE_LINKS)
    collection.add_links(
        [
            pystac.Link(
                "describedby",
                url,
                media_type=pystac.MediaType.HTML,
                title=f"OpenStreetMap wiki: {page}",
            )
            for page, url in dataset_wiki_pages(dataset_dir)
        ]
    )
    sources: list[OamImagery] = []
    all_classes = {"name": LABEL_CLASS_PROPERTY, "classes": sorted(stats["class_counts"])}
    collection.summaries.add("label:type", ["vector"])
    collection.summaries.add("label:tasks", tasks)
    collection.summaries.add("label:classes", [all_classes])
    collection.summaries.add("dataset:zoom", sorted({r["tile_z"] for r in records}))

    for project_id in sorted(project_names):
        project_records = [r for r in records if r["project_id"] == project_id]
        project_tiles = tiles[tiles["project_id"] == project_id]
        project_dir = stac_dir / f"project-{project_id}"
        imagery_dir = stac_dir / f"project-{project_id}-imagery"
        project_dir.mkdir()
        imagery_dir.mkdir()
        hrefs = {
            "imagery": imagery_dir / "imagery.tif",
            "labels": project_dir / "labels.geojson",
            "labels_parquet": project_dir / "labels.parquet",
            "tiles": project_dir / "tiles.parquet",
        }
        logger.info("STAC project %d: %d tiles", project_id, len(project_records))

        write_imagery_cog(project_records, hrefs["imagery"])
        project_labels = labels_frame(project_records)
        project_labels.to_file(hrefs["labels"], driver="GeoJSON")
        project_labels.to_parquet(hrefs["labels_parquet"])
        project_tiles.to_parquet(hrefs["tiles"])

        metadata = json.loads((dataset_dir / f"project_{project_id}" / "metadata.json").read_text())
        source = oam_imagery(metadata["imagery_url"], metadata["bbox"])
        sources.append(source)
        imagery_item = _imagery_item(project_records[0], project_tiles, source, hrefs["imagery"])
        label_item = _label_item(
            project_records[0],
            project_tiles,
            project_labels,
            extracted_at,
            hrefs,
            tasks,
            mapping_types[project_id],
        )
        label_item.add_link(
            pystac.Link(
                "source",
                imagery_item,
                media_type=pystac.MediaType.GEOJSON,
                title="Imagery the labels were drawn on",
                extra_fields={"label:assets": ["imagery"]},
            )
        )
        collection.add_items([label_item, imagery_item])

    first_capture = min(source.acquired for source in sources)
    collection.extent.temporal = pystac.TemporalExtent([[first_capture, extracted_at]])
    collection.normalize_hrefs(str(stac_dir))
    collection.make_all_asset_hrefs_relative()
    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    logger.info("STAC Collection written to %s", stac_dir / "collection.json")
    return stac_dir / "collection.json"


def write_published_stac(stac_dir: Path, base_url: str, out_dir: Path) -> None:
    """Write the catalogue JSON with absolute links under base_url, where it is published."""
    # STAC Browser loads a label item's source imagery without its address, so relative links fail
    base_url = base_url.rstrip("/")
    collection = pystac.Collection.from_file(str(stac_dir / "collection.json"))
    root = stac_dir.resolve()
    asset_paths = {
        (item.id, key): Path(str(asset.get_absolute_href())).relative_to(root).as_posix()
        for item in collection.get_items()
        for key, asset in item.assets.items()
    }
    collection.normalize_hrefs(base_url)
    for item in collection.get_items():
        for key, asset in item.assets.items():
            asset.href = f"{base_url}/{asset_paths[(item.id, key)]}"
    collection.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED, dest_href=str(out_dir))
