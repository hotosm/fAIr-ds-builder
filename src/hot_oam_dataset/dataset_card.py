"""Dataset card text from the dataset config, and statistics of the rows the card describes."""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import resources

from huggingface_hub import DatasetCard
from pyproj import Geod

CARD_TEMPLATE_FILENAME = "dataset_card.md"
CARD_DESCRIPTION_START = "<!-- DESCRIPTION_START -->"
CARD_DESCRIPTION_END = "<!-- DESCRIPTION_END -->"
CARD_REPO_PLACEHOLDER = "<repo>"
CARD_VERSION_PLACEHOLDER = "<version>"
CARD_SELECTION_PLACEHOLDER = "<selection>"
CARD_LIMITATIONS_PLACEHOLDER = "<limitations>"
CARD_AUTO_STATS_START = "<!-- AUTO_STATS_START -->"
CARD_AUTO_STATS_END = "<!-- AUTO_STATS_END -->"
DEFAULT_SAMPLE_TILES = 6
# STAC label extension tasks, with the Hugging Face task categories and ids each one implies
CARD_TASKS = {
    "segmentation": {
        "task_categories": ["image-segmentation"],
        "task_ids": ["semantic-segmentation"],
    },
    "detection": {"task_categories": ["object-detection"], "task_ids": []},
    "classification": {"task_categories": ["image-classification"], "task_ids": []},
}
# Ground metres per pixel of a 256-pixel Web Mercator tile at zoom 0 on the equator
EQUATOR_METRES_PER_PIXEL_ZOOM_0 = 156543.03392
WGS84 = Geod(ellps="WGS84")


def read_card_template() -> str:
    """The card layout shipped with the package; dataset text comes from the config."""
    return resources.files("hot_oam_dataset").joinpath(CARD_TEMPLATE_FILENAME).read_text()


def generic_card_template() -> str:
    """The card layout with the config-only selection and limitations left out."""
    text = read_card_template().replace(f"{CARD_SELECTION_PLACEHOLDER}\n\n", "")
    return text.replace(f"## Known limitations\n\n{CARD_LIMITATIONS_PLACEHOLDER}\n\n", "")


@dataclass(frozen=True)
class CardText:
    """The parts of a dataset card written by hand, read from the `card` key of a config."""

    repo_id: str
    title: str
    label_noun: str
    tasks: tuple[str, ...]
    summary: str
    selection: str
    limitations: tuple[str, ...]
    sample_tiles: int = DEFAULT_SAMPLE_TILES

    @classmethod
    def from_mapping(cls, mapping: dict) -> "CardText":
        expected = {
            "repo_id",
            "title",
            "label_noun",
            "tasks",
            "summary",
            "selection",
            "limitations",
        }
        optional = {"sample_tiles"}
        if not expected <= set(mapping) <= expected | optional:
            missing, unknown = expected - set(mapping), set(mapping) - expected - optional
            msg = f"card needs exactly {sorted(expected)}; missing {sorted(missing)}, "
            raise ValueError(msg + f"unknown {sorted(unknown)}")
        tasks = mapping["tasks"]
        if not isinstance(tasks, list) or not tasks or not set(tasks) <= set(CARD_TASKS):
            raise ValueError(f"card tasks must be a non-empty list from {sorted(CARD_TASKS)}")
        limitations = mapping["limitations"]
        if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
            raise ValueError("card limitations must be a list of strings")
        sample_tiles = mapping.get("sample_tiles", DEFAULT_SAMPLE_TILES)
        if not isinstance(sample_tiles, int) or isinstance(sample_tiles, bool) or sample_tiles < 0:
            raise ValueError("card sample_tiles must be a whole number, 0 to leave samples out")
        text_fields = expected - {"limitations", "tasks"}
        empty = sorted(key for key in text_fields if not str(mapping[key]).strip())
        if empty:
            raise ValueError(f"card fields must not be empty: {empty}")
        return cls(
            **{key: str(mapping[key]).strip() for key in text_fields},
            tasks=tuple(tasks),
            limitations=tuple(line.strip() for line in limitations),
            sample_tiles=sample_tiles,
        )


def _tile_area_sq_km(record: dict) -> float:
    west, south = record["bbox_west"], record["bbox_south"]
    east, north = record["bbox_east"], record["bbox_north"]
    area, _ = WGS84.polygon_area_perimeter([west, east, east, west], [south, south, north, north])
    return abs(area) / 1e6


def metres_per_pixel(latitude: float, zoom: int) -> float:
    return EQUATOR_METRES_PER_PIXEL_ZOOM_0 * math.cos(math.radians(latitude)) / 2**zoom


def published_stats(records: list[dict]) -> dict:
    """Counts over the rows written to the Parquet, after empty tiles are capped."""
    class_counts: Counter[str] = Counter()
    projects: dict[int, dict] = {}
    latitudes: dict[int, list[float]] = defaultdict(list)
    countries: dict[str, dict] = defaultdict(
        lambda: {"projects": set(), "tiles": 0, "labels": 0, "area_sq_km": 0.0}
    )

    for record in records:
        class_counts.update(json.loads(record["class_counts"]))
        project = projects.setdefault(
            record["project_id"],
            {
                "name": record["project_name"],
                "url": record["project_url"],
                "country": record["country"],
                "tiles": 0,
                "labels": 0,
                "zoom": record["tile_z"],
            },
        )
        project["tiles"] += 1
        project["labels"] += record["num_features"]
        latitudes[record["project_id"]].append((record["bbox_south"] + record["bbox_north"]) / 2)
        country = countries[record["country"]]
        country["projects"].add(record["project_id"])
        country["tiles"] += 1
        country["labels"] += record["num_features"]
        country["area_sq_km"] += _tile_area_sq_km(record)

    for project_id, project in projects.items():
        mean_latitude = sum(latitudes[project_id]) / len(latitudes[project_id])
        project["metres_per_pixel"] = metres_per_pixel(mean_latitude, project["zoom"])

    labels = sum(record["num_features"] for record in records)
    tiles_with_labels = sum(1 for record in records if record["num_features"] > 0)
    return {
        "tiles": len(records),
        "tiles_with_labels": tiles_with_labels,
        "tiles_without_labels": len(records) - tiles_with_labels,
        "labels": labels,
        "labels_per_tile": labels / len(records) if records else 0.0,
        "area_sq_km": sum(country["area_sq_km"] for country in countries.values()),
        "zoom_levels": sorted({record["tile_z"] for record in records}),
        "class_counts": dict(class_counts.most_common()),
        "projects": dict(sorted(projects.items())),
        "countries": {
            name: {**info, "projects": len(info["projects"])}
            for name, info in sorted(countries.items(), key=lambda item: -item[1]["tiles"])
        },
    }


def source_project_lines(stats: dict, noun: str) -> list[str]:
    lines = [
        f"| Project | Country | Tiles | {noun.capitalize()} | Ground pixel size |",
        "|---------|---------|-------|------|-------------------|",
    ]
    for project_id, project in stats["projects"].items():
        lines.append(
            f"| [{project_id}: {project['name']}]({project['url']}) | {project['country']} "
            f"| {project['tiles']:,} | {project['labels']:,} "
            f"| {project['metres_per_pixel']:.2f} m |"
        )
    return lines


def _description(card_text: CardText) -> str:
    return "\n".join(
        [
            CARD_DESCRIPTION_START,
            f"# {card_text.title}",
            "",
            card_text.summary,
            "",
            CARD_AUTO_STATS_START,
            CARD_AUTO_STATS_END,
            CARD_DESCRIPTION_END,
        ]
    )


def render_card_text(template: str, card_text: CardText, dataset_version: str) -> str:
    """Fill the template's description block, repo id and task from the config's card text."""
    pattern = re.compile(
        rf"{re.escape(CARD_DESCRIPTION_START)}.*?{re.escape(CARD_DESCRIPTION_END)}",
        flags=re.DOTALL,
    )
    if len(pattern.findall(template)) != 1:
        msg = f"card template needs one {CARD_DESCRIPTION_START} ... {CARD_DESCRIPTION_END} block"
        raise ValueError(msg)
    text = pattern.sub(lambda _: _description(card_text), template)
    text = text.replace(CARD_SELECTION_PLACEHOLDER, card_text.selection)
    text = text.replace(
        CARD_LIMITATIONS_PLACEHOLDER, "\n".join(f"- {line}" for line in card_text.limitations)
    )
    text = text.replace(CARD_REPO_PLACEHOLDER, card_text.repo_id)
    text = text.replace(CARD_VERSION_PLACEHOLDER, dataset_version)

    card = DatasetCard(text)
    card.data["pretty_name"] = card_text.title
    for key in ("task_categories", "task_ids"):
        values = [value for task in card_text.tasks for value in CARD_TASKS[task][key]]
        card.data[key] = values or None
    return str(card)
