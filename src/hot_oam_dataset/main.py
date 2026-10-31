import argparse
import dataclasses
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from hot_oam_dataset import __version__
from hot_oam_dataset.dataset_builder import (
    MINIMUM_VALID_CHIP_BYTES,
    DatasetConfig,
    build_all_projects,
)
from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.fair_catalog import FAIR_API_KEY_ENV, FairSettings, publish_to_fair
from hot_oam_dataset.feature_spec import (
    DEFAULT_FEATURE_TYPE,
    GEOMETRY_TYPES,
    TM_MAPPING_TYPES,
    resolve_feature_spec,
)
from hot_oam_dataset.hf_publisher import (
    compress_projects,
    generate_parquet,
    push_raw_to_hub,
    push_to_hub,
)
from hot_oam_dataset.root_catalog import update_root_catalog
from hot_oam_dataset.split_generator import SplitSettings, generate_splits
from hot_oam_dataset.stac_export import generate_stac
from hot_oam_dataset.tm_client import (
    discover_oam_projects,
    fetch_projects_by_ids,
)

logger = logging.getLogger(__name__)

CARD_CONFIG_KEY = "card"
FAIR_CONFIG_KEY = "fair"
SPLIT_CONFIG_KEY = "split"


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {number}")
    return number


def _comma_separated_ints(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("must contain at least one project id")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be comma-separated integers ({exc})") from exc


def _load_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} must hold a mapping of option names to values")
    return config


def _config_file_arguments(config: dict) -> list[str]:
    """Turn the YAML options into flags, so argparse validates them like the command line."""
    arguments: list[str] = []
    for key, value in config.items():
        if key in (CARD_CONFIG_KEY, SPLIT_CONFIG_KEY, FAIR_CONFIG_KEY):
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            arguments.append(flag)
        elif value is False:
            arguments.append("--no-" + flag.removeprefix("--"))
        elif isinstance(value, list):
            arguments.extend([flag, *map(str, value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    hf_repo_id_from_env = os.getenv("HF_DATASET_REPO_ID", "")
    feature_type_from_env = os.getenv("HOT_OAM_FEATURE_TYPE", DEFAULT_FEATURE_TYPE)
    burn_binary_masks_env = os.getenv("HOT_OAM_BURN_BINARY_MASKS", "true").strip().lower()
    burn_binary_masks_default = burn_binary_masks_env in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(
        description=(
            "Build segmentation and object detection datasets from HOT Tasking Manager "
            "OAM projects, labelled with any OpenStreetMap feature type"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="YAML file of option values for one dataset; command-line flags override it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Root output directory for the dataset (default: dataset)",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="Number of years to look back for projects (default: 5)",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Tile zoom level (default: 19)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max projects to discover and download (default: 5, use 0 for all)",
    )
    parser.add_argument(
        "--project-ids",
        type=_comma_separated_ints,
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated TM project IDs to build (e.g. 25317,25320). "
            "Bypasses discovery: --years-back and --limit are ignored. "
            "Validation and OAM-imagery checks are downgraded to warnings"
        ),
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover and list projects without downloading data",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="After the build, generate the splits, Parquet, dataset card and STAC catalogue",
    )
    parser.add_argument(
        "--publish-stac",
        metavar="STAC_API_URL",
        help=(
            "Register the dataset in a fAIr STAC API, e.g. https://stac.ai.hotosm.org/stac, "
            f"after --push-to-hub; needs the config's card and fair sections and {FAIR_API_KEY_ENV}"
        ),
    )
    parser.add_argument(
        "--update-root-catalog",
        metavar="ROOT_REPO_ID",
        help=(
            "Link the dataset's Hub release as a child of the STAC catalogue in this Hub repo, "
            "e.g. hotosm/fAIr-guru-stac, after --push-to-hub; needs the config's card section"
        ),
    )
    parser.add_argument(
        "--generate-parquet",
        action="store_true",
        help="Generate HuggingFace-compatible Parquet from existing dataset",
    )
    parser.add_argument(
        "--generate-splits",
        action="store_true",
        help=(
            "Assign contiguous regions of each project to train, validation or test, as set "
            "by the config's split section. Writes splits.json to the output directory"
        ),
    )
    parser.add_argument(
        "--generate-stac",
        action="store_true",
        help=(
            "Write a static STAC Collection under stac/ from the split Parquets: per-project "
            "imagery COG, GeoParquet labels and tile footprints, and a split map"
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        nargs="?",
        const=hf_repo_id_from_env,
        default=None,
        metavar="REPO_ID",
        help=(
            "Push dataset to HuggingFace Hub (e.g. hotosm/vhr-building-segmentation). "
            "If omitted, uses HF_DATASET_REPO_ID environment variable"
        ),
    )
    parser.add_argument(
        "--push-raw",
        type=str,
        nargs="?",
        const=hf_repo_id_from_env,
        default=None,
        metavar="REPO_ID",
        help=(
            "Upload raw chips/labels/metadata to HuggingFace Hub under raw/. "
            "If omitted, uses HF_DATASET_REPO_ID environment variable"
        ),
    )
    parser.add_argument(
        "--compress-raw",
        action="store_true",
        help=(
            "Compress chips/masks/labels into data.tar.gz per project before uploading. "
            "Can be used standalone or combined with --push-raw"
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=__version__,
        help=f"Dataset version embedded in stats and Parquet metadata (default: {__version__})",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default=feature_type_from_env,
        help=(
            "OSM tag key to label with, e.g. building, highway, waterway, landuse "
            f"(default: HOT_OAM_FEATURE_TYPE env or {DEFAULT_FEATURE_TYPE}). "
            "Any OSM key works; unknown keys discover against the OTHER mapping type"
        ),
    )
    parser.add_argument(
        "--tm-mapping-type",
        type=str,
        choices=TM_MAPPING_TYPES,
        default=None,
        help=(
            "Tasking Manager mapping type used for project discovery. "
            "Defaults to the one implied by --feature-type"
        ),
    )
    parser.add_argument(
        "--geometry-types",
        type=str,
        nargs="+",
        choices=GEOMETRY_TYPES,
        default=None,
        metavar="TYPE",
        help=(
            "Geometry filter sent to the OSM Raw Data API. "
            "Defaults to the one implied by --feature-type (polygon for areas, line for roads)"
        ),
    )
    parser.add_argument(
        "--tag-values",
        type=str,
        nargs="+",
        default=None,
        metavar="VALUE",
        help=(
            "Keep only these values of the --feature-type tag, e.g. "
            "--feature-type natural --tag-values tree tree_group (default: any value)"
        ),
    )
    parser.add_argument(
        "--require-tags",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY",
        help=(
            "Keep only features that also carry these tag keys, any value, e.g. "
            "--feature-type surface --require-tags highway (default: none)"
        ),
    )
    parser.add_argument(
        "--negative-ratio",
        type=_non_negative_float,
        default=None,
        metavar="RATIO",
        help=(
            "Cap empty tiles in the Parquet at RATIO per labelled tile, per project "
            "(e.g. 1.0 for a 1:1 balance). Labelled tiles are never dropped. "
            "Applies to --generate-parquet and --push-to-hub; raw files keep every tile"
        ),
    )
    parser.add_argument(
        "--imagery-url",
        default=None,
        metavar="TMS",
        help=(
            "Tile URL ({z}/{x}/{y}, 256 px tiles) for every project's imagery, used in place of "
            "the Tasking Manager imagery URL (default: each project's Tasking Manager imagery)"
        ),
    )
    parser.add_argument(
        "--label-files",
        default=None,
        metavar="DIR_OR_URL",
        help=(
            "Folder or URL holding project_<id>_labels.geojson and project_<id>_coverage.geojson. "
            "Labels are read from these files in place of OpenStreetMap, and tiles are kept only "
            "inside the coverage (default: OpenStreetMap through the HOT Raw Data API)"
        ),
    )
    parser.add_argument(
        "--label-buffer-meters",
        type=float,
        default=0.0,
        help=(
            "Widen label geometries by this many metres before burning masks. "
            "Line features need a buffer to produce usable masks (default: 0)"
        ),
    )
    parser.add_argument(
        "--drop-small-chips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            f"Remove chips under {MINIMUM_VALID_CHIP_BYTES:,} bytes, which are usually failed "
            "downloads. Turn off for "
            "imagery whose valid tiles compress below that, such as calm water (default: on)"
        ),
    )
    parser.add_argument(
        "--burn-binary-masks",
        action=argparse.BooleanOptionalAction,
        default=burn_binary_masks_default,
        help=(
            "Generate binary raster masks from OSM labels for each project "
            "(default: enabled; can be controlled by HOT_OAM_BURN_BINARY_MASKS)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    argv = sys.argv[1:] if argv is None else argv
    config_path = parser.parse_known_args(argv)[0].config
    if config_path and not config_path.is_file():
        parser.error(f"config file {config_path} does not exist")
    config = _load_config(config_path) if config_path else {}
    args = parser.parse_args(_config_file_arguments(config) + argv)
    card = config.get(CARD_CONFIG_KEY)
    # Reported like any other config error, as a usage message naming the file
    try:
        args.card_text = CardText.from_mapping(card) if card is not None else None
        args.split_settings = SplitSettings.from_mapping(config.get(SPLIT_CONFIG_KEY) or {})
        fair = config.get(FAIR_CONFIG_KEY)
        args.fair_settings = FairSettings.from_mapping(fair) if fair is not None else None
    except (ValueError, TypeError) as exc:
        parser.error(f"{config_path}: {exc}")
    single_steps = [
        flag
        for flag, chosen in (
            ("--discover-only", args.discover_only),
            ("--generate-splits", args.generate_splits),
            ("--generate-parquet", args.generate_parquet),
            ("--generate-stac", args.generate_stac),
            ("--push-to-hub", args.push_to_hub is not None),
            ("--push-raw", args.push_raw is not None),
            ("--publish-stac", args.publish_stac),
            ("--update-root-catalog", args.update_root_catalog),
            ("--compress-raw", args.compress_raw),
        )
        if chosen
    ]
    if args.all and single_steps:
        parser.error(f"--all runs every build step and cannot be combined with {single_steps[0]}")
    if args.publish_stac and len(single_steps) > 1:
        parser.error("--publish-stac runs on its own, after --push-to-hub has finished")
    if args.update_root_catalog and len(single_steps) > 1:
        parser.error("--update-root-catalog runs on its own, after --push-to-hub has finished")
    return args


def _generate_parquet(args, output_dir: Path) -> None:
    generate_parquet(
        output_dir,
        dataset_version=args.dataset_version,
        negative_ratio=args.negative_ratio,
        card_text=args.card_text,
    )


def _generate_stac(args, output_dir: Path) -> None:
    generate_stac(output_dir, dataset_version=args.dataset_version, card_text=args.card_text)


def _run_standalone_command(args, output_dir: Path) -> bool:
    if args.generate_splits:
        generate_splits(output_dir, args.split_settings)
        return True

    if args.generate_parquet:
        _generate_parquet(args, output_dir)
        return True

    if args.generate_stac:
        _generate_stac(args, output_dir)
        return True

    if args.push_to_hub is not None:
        _require_repo_id(args.push_to_hub, "--push-to-hub")
        push_to_hub(
            output_dir,
            args.push_to_hub,
            dataset_version=args.dataset_version,
            negative_ratio=args.negative_ratio,
            card_text=args.card_text,
        )
        return True

    if args.publish_stac:
        if not args.card_text or not args.fair_settings:
            logger.error("--publish-stac needs a config with card and fair sections")
            sys.exit(2)
        publish_to_fair(
            output_dir,
            args.publish_stac,
            args.card_text,
            args.fair_settings,
            args.dataset_version,
        )
        return True

    if args.update_root_catalog:
        if not args.card_text:
            logger.error("--update-root-catalog needs a config with a card section")
            sys.exit(2)
        update_root_catalog(args.update_root_catalog, args.card_text, args.dataset_version)
        return True

    if args.push_raw is not None:
        _require_repo_id(args.push_raw, "--push-raw")
        push_raw_to_hub(
            output_dir,
            args.push_raw,
            dataset_version=args.dataset_version,
            compress=args.compress_raw,
        )
        return True

    if args.compress_raw:
        compress_projects(output_dir)
        return True

    return False


def _require_repo_id(repo_id: str, flag: str) -> None:
    if not repo_id:
        logger.error("REPO_ID missing. Pass %s <repo_id> or set HF_DATASET_REPO_ID", flag)
        sys.exit(2)


def _collect_projects(args, spec, output_dir: Path) -> list:
    if args.project_ids:
        logger.info(
            "Building hand-picked projects %s (ignoring --years-back and --limit)",
            args.project_ids,
        )
        return fetch_projects_by_ids(
            args.project_ids,
            mapping_type=spec.tm_mapping_type,
            output_path=output_dir,
        )

    now = datetime.now(tz=UTC)
    created_from = now.replace(year=now.year - args.years_back).strftime("%Y-%m-%d")
    created_to = now.strftime("%Y-%m-%d")

    logger.info("Searching TM projects from %s to %s", created_from, created_to)

    return discover_oam_projects(
        mapping_type=spec.tm_mapping_type,
        created_from=created_from,
        created_to=created_to,
        output_path=output_dir,
        limit=args.limit or None,
    )


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if _run_standalone_command(args, output_dir):
        return

    spec = resolve_feature_spec(
        args.feature_type,
        args.tm_mapping_type,
        args.geometry_types,
        args.tag_values,
        args.require_tags,
    )
    logger.info(
        "Feature type '%s' -> TM mapping type %s, geometry %s",
        spec.osm_tag,
        spec.tm_mapping_type,
        "/".join(spec.geometry_types),
    )

    projects = _collect_projects(args, spec, output_dir)
    if args.imagery_url:
        projects = [
            dataclasses.replace(project, imagery_url=args.imagery_url) for project in projects
        ]

    if not projects:
        logger.warning("No projects to build")
        sys.exit(0)

    _print_projects_table(projects)

    if args.discover_only:
        logger.info("Discovery complete. Saved to %s", output_dir)
        return

    config = DatasetConfig(
        output_dir=output_dir,
        zoom=args.zoom,
        feature_type=args.feature_type,
        burn_binary_masks=args.burn_binary_masks,
        label_buffer_meters=args.label_buffer_meters,
        tm_mapping_type=args.tm_mapping_type,
        geometry_types=args.geometry_types,
        tag_values=args.tag_values,
        required_tags=args.require_tags,
        label_files=args.label_files,
        drop_small_chips=args.drop_small_chips,
    )
    failed_project_ids = build_all_projects(projects, config, dataset_version=args.dataset_version)
    if failed_project_ids:
        sys.exit(1)

    if args.all:
        generate_splits(output_dir, args.split_settings)
        _generate_parquet(args, output_dir)
        _generate_stac(args, output_dir)


def _truncate(text: str, max_length: int) -> str:
    return text[:max_length] if len(text) > max_length else text


def _print_projects_table(projects: list) -> None:
    header = f"{'ID':<8} {'Name':<50} {'Validated%':<12} {'Country':<20} {'Org':<25}"
    separator = "-" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)
    for p in projects:
        name = _truncate(p.name, 48)
        country = _truncate(p.country, 18)
        org = _truncate(p.organisation, 23)
        print(f"{p.project_id:<8} {name:<50} {p.percent_validated:<12} {country:<20} {org:<25}")
    print(separator)
    print(f"Total: {len(projects)} projects\n")


if __name__ == "__main__":
    main()
