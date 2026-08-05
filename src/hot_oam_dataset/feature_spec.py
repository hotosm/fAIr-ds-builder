from dataclasses import dataclass

TM_MAPPING_TYPES = ("BUILDINGS", "ROADS", "WATERWAYS", "LAND_USE", "OTHER")
GEOMETRY_TYPES = ("point", "line", "polygon", "all_geometry")

DEFAULT_FEATURE_TYPE = "building"
FALLBACK_MAPPING_TYPE = "OTHER"
FALLBACK_GEOMETRY_TYPES = ("point", "line", "polygon")


@dataclass(frozen=True)
class FeatureSpec:
    """Maps an OSM tag key onto Tasking Manager discovery, the Raw Data API, and report wording."""

    osm_tag: str
    tm_mapping_type: str
    geometry_types: tuple[str, ...]
    label_noun: str


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
) -> FeatureSpec:
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
    )
