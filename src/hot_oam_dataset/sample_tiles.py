"""A grid of labelled tiles, each shown beside a plot of its labels coloured by class."""

import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import rasterio
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from shapely.geometry import shape

SAMPLE_TILES_FILENAME = "samples.png"
TILES_PER_ROW = 3
PANEL_INCHES = 2.2
CLASS_COLORS = ["#ffffff", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2"]
OTHER_CLASS_COLOR = "#7f7f7f"
PROMINENT_CANDIDATES = 10
MIN_CLASS_SHARE = 0.01


def _class_counts(record: dict) -> dict[str, int]:
    return json.loads(record["class_counts"])


def _label_prominence(record: dict) -> tuple[int, float]:
    features = json.loads(record["label_geojson"])["features"]
    return record["num_features"], sum(shape(f["geometry"]).area for f in features)


def pick_sample_tiles(records: Sequence[dict], count: int, seed: int) -> list[tuple[dict, str]]:
    """Tiles with classified labels, each paired with the class it shows, common classes first."""
    labelled = sorted((r for r in records if _class_counts(r)), key=lambda r: r["tile_id"])
    generator = random.Random(seed)
    class_totals: Counter[str] = Counter()
    for record in labelled:
        class_totals.update(_class_counts(record))

    total = sum(class_totals.values())
    common = [
        name for name, value in class_totals.most_common() if value >= MIN_CLASS_SHARE * total
    ]
    picked: list[tuple[dict, str]] = []
    if len(common) > 1:
        for name in common[:count]:
            chosen = [record for record, _ in picked]
            candidates = sorted(
                (r for r in labelled if name in _class_counts(r) and r not in chosen),
                key=lambda r: -_class_counts(r)[name],
            )
            # Among the tiles richest in the class, so it is visible in the sample
            if candidates:
                picked.append((generator.choice(candidates[:PROMINENT_CANDIDATES]), name))
    chosen = [record for record, _ in picked]
    # Filled from the quarter of tiles with the most, then largest, labels so the labels show
    remaining = sorted(
        (r for r in labelled if r not in chosen), key=_label_prominence, reverse=True
    )
    remaining = remaining[: max(len(remaining) // 4, count)]
    fill = generator.sample(remaining, min(count - len(picked), len(remaining)))
    picked.extend((record, Counter(_class_counts(record)).most_common(1)[0][0]) for record in fill)
    return picked


def _draw_pair(
    image_axis, mask_axis, record: dict, shown_class: str, feature_type: str, colors: dict
) -> None:
    bounds = (record["bbox_west"], record["bbox_east"], record["bbox_south"], record["bbox_north"])
    with rasterio.open(record["image"]) as chip:
        pixels = chip.read([1, 2, 3]).transpose(1, 2, 0)
    image_axis.imshow(pixels, extent=bounds)

    labels = json.loads(record["label_geojson"])
    features = gpd.GeoDataFrame.from_features(labels["features"], crs="EPSG:4326")
    features["color"] = [
        colors.get(str((tags or {}).get(feature_type)), OTHER_CLASS_COLOR)
        for tags in (feature["properties"].get("tags") for feature in labels["features"])
    ]
    mask_axis.set_facecolor("black")
    features.plot(ax=mask_axis, color=features["color"], linewidth=1.5, markersize=8)

    image_axis.set_title(f"{shown_class}: {_class_counts(record)[shown_class]}", fontsize=8)
    # Degrees stretched by latitude, so the tiles stay square as on a Web Mercator map
    aspect = 1 / math.cos(math.radians((bounds[2] + bounds[3]) / 2))
    for axis in (image_axis, mask_axis):
        axis.set_aspect(aspect)
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_xticks([])
        axis.set_yticks([])


def write_sample_tiles(
    records: Sequence[dict], feature_type: str, count: int, seed: int, path: Path
) -> bool:
    """Draw up to count sample tiles to path; False when the dataset has no labelled tile."""
    samples = pick_sample_tiles(records, count, seed)
    if not samples:
        return False

    shown: Counter[str] = Counter()
    for record, _ in samples:
        shown.update(_class_counts(record))
    featured = list(dict.fromkeys(name for _, name in samples))
    ordered = featured + [name for name, _ in shown.most_common() if name not in featured]
    colors = dict(zip(ordered, CLASS_COLORS, strict=False))
    rows = -(-len(samples) // TILES_PER_ROW)
    columns = 2 * min(TILES_PER_ROW, len(samples))
    figure = Figure(
        figsize=(PANEL_INCHES * columns, PANEL_INCHES * rows + 0.5), layout="constrained"
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for index, (record, shown_class) in enumerate(samples):
        row, column = divmod(index, TILES_PER_ROW)
        image_axis, mask_axis = axes[row][2 * column], axes[row][2 * column + 1]
        _draw_pair(image_axis, mask_axis, record, shown_class, feature_type, colors)
    for axis in axes.flatten()[2 * len(samples) :]:
        axis.set_axis_off()

    handles = [Patch(color=color, label=name) for name, color in colors.items()]
    if len(shown) > len(colors):
        handles.append(Patch(color=OTHER_CLASS_COLOR, label="other"))
    if len(handles) > 1:
        figure.legend(
            handles=handles,
            loc="outside lower center",
            ncols=min(len(handles), 8),
            facecolor="#dddddd",
        )
    figure.savefig(path, dpi=110)
    return True
