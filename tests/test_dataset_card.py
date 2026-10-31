import json

import pytest
from huggingface_hub import DatasetCard

from hot_oam_dataset.dataset_card import (
    CardText,
    published_stats,
    read_card_template,
    render_card_text,
    source_project_lines,
)

CARD_MAPPING = {
    "repo_id": "hotosm/fAIr-guru-td-trees",
    "title": "fAIr Open Mapping Gurus: trees",
    "label_noun": "trees",
    "tasks": ["segmentation", "classification"],
    "summary": "Tree training data.",
    "selection": "Trees mapped as points.",
    "limitations": ["Crown diameters are estimates."],
}

TEMPLATE = """\
---
pretty_name: HOT OAM Dataset
task_categories:
  - image-segmentation
task_ids:
  - semantic-segmentation
---

<!-- DESCRIPTION_START -->
# HOT OAM Dataset
<!-- DESCRIPTION_END -->

## How it was labelled

<selection>

## Known limitations

<limitations>

## Load

load_dataset("<repo>")

![Split map](https://huggingface.co/datasets/<repo>/resolve/v<version>/stac/split_map.png)
"""


def _row(latitude=0.0, size=0.01, zoom=19, num_features=1, project_id=1):
    return {
        "project_id": project_id,
        "project_name": "Freetown trees",
        "project_url": f"https://tasks.hotosm.org/projects/{project_id}",
        "country": "Sierra Leone",
        "tile_z": zoom,
        "num_features": num_features,
        "class_counts": json.dumps({"tree": num_features}),
        "bbox_west": 0.0,
        "bbox_south": latitude,
        "bbox_east": size,
        "bbox_north": latitude + size,
    }


def test_card_text_rejects_missing_unknown_and_invalid_fields():
    misspelt = {key: value for key, value in CARD_MAPPING.items() if key != "title"}
    misspelt["titel"] = CARD_MAPPING["title"]
    with pytest.raises(ValueError, match=r"missing \['title'\], unknown \['titel'\]"):
        CardText.from_mapping(misspelt)
    with pytest.raises(ValueError, match="tasks must be a non-empty list"):
        CardText.from_mapping({**CARD_MAPPING, "tasks": ["regression"]})
    with pytest.raises(ValueError, match="list of strings"):
        CardText.from_mapping({**CARD_MAPPING, "limitations": "one line"})
    with pytest.raises(ValueError, match=r"must not be empty: \['summary'\]"):
        CardText.from_mapping({**CARD_MAPPING, "summary": "  "})


def test_tile_area_is_the_geodesic_footprint():
    stats = published_stats([_row(size=0.01)])

    # 0.01 degrees spans 1,105.7 m of latitude and 1,113.2 m of longitude at the equator
    assert stats["area_sq_km"] == pytest.approx(1.2309, rel=1e-3)


@pytest.mark.parametrize(
    ("latitude", "zoom", "metres"), [(0.0, 19, 0.2986), (-33.9, 19, 0.2479), (8.45, 20, 0.1477)]
)
def test_ground_pixel_size_follows_zoom_and_latitude(latitude, zoom, metres):
    stats = published_stats([_row(latitude=latitude, size=0.0001, zoom=zoom)])

    assert stats["projects"][1]["metres_per_pixel"] == pytest.approx(metres, rel=1e-3)


def test_rendered_card_carries_the_config_text_repo_and_task():
    card_text = CardText.from_mapping(CARD_MAPPING)

    rendered = render_card_text(TEMPLATE, card_text, "0.1.0")

    card = DatasetCard(rendered)
    assert card.data["pretty_name"] == "fAIr Open Mapping Gurus: trees"
    assert card.data["task_categories"] == ["image-segmentation", "image-classification"]
    assert card.data["task_ids"] == ["semantic-segmentation"]
    assert "# HOT OAM Dataset" not in rendered
    assert "# fAIr Open Mapping Gurus: trees\n\nTree training data." in rendered
    assert "## How it was labelled\n\nTrees mapped as points." in rendered
    assert "## Known limitations\n\n- Crown diameters are estimates." in rendered
    assert "<!-- AUTO_STATS_START -->\n<!-- AUTO_STATS_END -->" in rendered
    assert 'load_dataset("hotosm/fAIr-guru-td-trees")' in rendered
    assert "fAIr-guru-td-trees/resolve/v0.1.0/stac/split_map.png" in rendered


def test_card_text_sample_tiles_defaults_and_rejects_invalid_counts():
    assert CardText.from_mapping(CARD_MAPPING).sample_tiles == 6
    assert CardText.from_mapping({**CARD_MAPPING, "sample_tiles": 0}).sample_tiles == 0
    for invalid in (-1, "six", True):
        with pytest.raises(ValueError, match="sample_tiles"):
            CardText.from_mapping({**CARD_MAPPING, "sample_tiles": invalid})


def test_source_projects_table_names_and_links_each_project():
    stats = published_stats([_row(num_features=3), _row(num_features=0)])

    lines = source_project_lines(stats, "trees")

    assert lines[0] == "| Project | Country | Tiles | Trees | Ground pixel size |"
    project = "| [1: Freetown trees](https://tasks.hotosm.org/projects/1) | Sierra Leone | 2 | 3 |"
    assert lines[2].startswith(project)


def test_rendering_needs_one_description_block():
    card_text = CardText.from_mapping(CARD_MAPPING)

    with pytest.raises(ValueError, match="needs one"):
        render_card_text("# No markers\n", card_text, "0.1.0")


def test_packaged_card_links_the_release_in_stac_browser():
    card_text = CardText.from_mapping(CARD_MAPPING)

    rendered = render_card_text(read_card_template(), card_text, "0.1.0")

    browser = "https://radiantearth.github.io/stac-browser/#/external/huggingface.co/datasets"
    assert f"{browser}/hotosm/fAIr-guru-td-trees/raw/v0.1.0/stac/collection.json" in rendered
    assert "<repo>" not in rendered
    assert "<version>" not in rendered
    assert "<selection>" not in rendered
    assert "<limitations>" not in rendered


def test_card_label_source_replaces_openstreetmap_as_the_source_of_the_labels():
    sandbox = {"name": "the OSM Sandbox", "attribution": "HOT OpenStreetMap sandbox contributors"}
    card_text = CardText.from_mapping(CARD_MAPPING | {"label_source": sandbox})

    card = render_card_text(read_card_template(), card_text, "0.1.0")

    assert "The labels are the OSM Sandbox features" in card
    assert (
        "| Labels | the OSM Sandbox | ODbL 1.0, attribute HOT OpenStreetMap sandbox contributors |"
        in card
    )
    assert "Organised Editing Guidelines" not in card
    assert "[OpenStreetMap](https://www.openstreetmap.org/)" not in card


def test_card_without_label_source_names_openstreetmap():
    card = render_card_text(read_card_template(), CardText.from_mapping(CARD_MAPPING), "0.1.0")

    assert "The labels are [OpenStreetMap](https://www.openstreetmap.org/) features" in card
    assert "Organised Editing Guidelines" in card


@pytest.mark.parametrize(
    "label_source", [{"name": "Sandbox"}, {"name": "", "attribution": "HOT"}, "Sandbox"]
)
def test_card_label_source_needs_a_name_and_attribution(label_source):
    with pytest.raises(ValueError, match="label_source"):
        CardText.from_mapping(CARD_MAPPING | {"label_source": label_source})


def test_detection_card_drops_the_template_task_ids():
    card_text = CardText.from_mapping({**CARD_MAPPING, "tasks": ["detection"]})

    rendered = render_card_text(TEMPLATE, card_text, "0.1.0")

    assert DatasetCard(rendered).data["task_categories"] == ["object-detection"]
    assert "task_ids" not in rendered
