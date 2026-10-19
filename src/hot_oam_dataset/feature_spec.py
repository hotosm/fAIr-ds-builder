import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TM_MAPPING_TYPES = ("BUILDINGS", "ROADS", "WATERWAYS", "LAND_USE", "OTHER")
GEOMETRY_TYPES = ("point", "line", "polygon", "all_geometry")

DEFAULT_FEATURE_TYPE = "building"
FALLBACK_MAPPING_TYPE = "OTHER"
FALLBACK_GEOMETRY_TYPES = ("point", "line", "polygon")
OSM_WIKI_URL = "https://wiki.openstreetmap.org/wiki/"


@dataclass(frozen=True)
class FeatureSpec:
    """Maps an OSM tag key onto Tasking Manager discovery, the Raw Data API, and report wording."""

    osm_tag: str
    tm_mapping_type: str
    geometry_types: tuple[str, ...]
    label_noun: str
    tag_values: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()

    def raw_data_filters(self) -> dict[str, dict]:
        """Raw Data API filter for the tag, ANDed with any required keys; [] matches any value."""
        tags = {self.osm_tag: list(self.tag_values)}
        if not self.required_tags:
            return {"tags": {"all_geometry": {"join_or": tags}}}
        required = {key: [] for key in self.required_tags}
        return {"tags": {"all_geometry": {"join_and": tags | required}}}


KNOWN_FEATURE_SPECS = {
    spec.osm_tag: spec
    for spec in (
        FeatureSpec("building", "BUILDINGS", ("polygon",), "buildings"),
        FeatureSpec("highway", "ROADS", ("line",), "roads"),
        FeatureSpec("waterway", "WATERWAYS", ("line",), "waterways"),
        FeatureSpec("landuse", "LAND_USE", ("polygon",), "landuse areas"),
    )
}


def resolve_feature_spec(
    osm_tag: str,
    tm_mapping_type: str | None = None,
    geometry_types: list[str] | tuple[str, ...] | None = None,
    tag_values: list[str] | tuple[str, ...] | None = None,
    required_tags: list[str] | tuple[str, ...] | None = None,
) -> FeatureSpec:
    # The Raw Data API interpolates tag values and keys into SQL
    invalid_values = [value for value in tag_values or () if not value or "'" in value]
    if invalid_values:
        raise ValueError(f"Tag values must be non-empty and free of quotes: {invalid_values}")
    invalid_keys = [key for key in required_tags or () if not key or "'" in key or key == osm_tag]
    if invalid_keys:
        raise ValueError(
            f"Required tags must be non-empty, free of quotes and differ from {osm_tag}: "
            f"{invalid_keys}"
        )

    defaults = KNOWN_FEATURE_SPECS.get(osm_tag)
    if defaults is None:
        defaults = FeatureSpec(
            osm_tag=osm_tag,
            tm_mapping_type=FALLBACK_MAPPING_TYPE,
            geometry_types=FALLBACK_GEOMETRY_TYPES,
            label_noun=f"{osm_tag} features",
        )

    return FeatureSpec(
        osm_tag=defaults.osm_tag,
        tm_mapping_type=tm_mapping_type or defaults.tm_mapping_type,
        geometry_types=tuple(geometry_types) if geometry_types else defaults.geometry_types,
        label_noun=defaults.label_noun,
        tag_values=tuple(tag_values or ()),
        required_tags=tuple(required_tags or ()),
    )


def label_class_counts(label_data: dict, feature_type: str) -> Counter[str]:
    """Count values of the feature_type tag, e.g. how many residential vs path roads."""
    counts: Counter[str] = Counter()
    for feature in label_data.get("features", []):
        tags = (feature.get("properties") or {}).get("tags") or {}
        value = tags.get(feature_type)
        if value:
            counts[value] += 1
    return counts


def osm_wiki_pages(
    osm_tag: str, tag_values: Sequence[str], required_tags: Sequence[str]
) -> list[tuple[str, str]]:
    """OpenStreetMap wiki pages defining the tags a dataset's labels were selected on."""
    pages = [f"Key:{osm_tag}", *(f"Tag:{osm_tag}={value}" for value in tag_values)]
    pages += [f"Key:{key}" for key in required_tags]
    return [(page, f"{OSM_WIKI_URL}{page}") for page in pages]


def dataset_wiki_pages(dataset_dir: Path) -> list[tuple[str, str]]:
    """Wiki pages for the tags every project of a built dataset was labelled with."""
    pages: dict[str, str] = {}
    for metadata_path in sorted(dataset_dir.glob("project_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        for page, url in osm_wiki_pages(
            metadata["feature_type"], metadata["tag_values"], metadata["required_tags"]
        ):
            pages.setdefault(page, url)
    return list(pages.items())
