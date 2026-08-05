import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from hot_oam_dataset import __version__
from hot_oam_dataset.dataset_builder import DatasetConfig, build_all_projects
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
from hot_oam_dataset.split_generator import generate_splits
from hot_oam_dataset.tm_client import (
    discover_oam_projects,
    fetch_projects_by_ids,
)

logger = logging.getLogger(__name__)


def _comma_separated_ints(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("must contain at least one project id")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be comma-separated integers ({exc})") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    hf_repo_id_from_env = os.getenv("HF_DATASET_REPO_ID", "")
    feature_type_from_env = os.getenv("HOT_OAM_FEATURE_TYPE", DEFAULT_FEATURE_TYPE)
    burn_binary_masks_env = os.getenv("HOT_OAM_BURN_BINARY_MASKS", "true").strip().lower()
    burn_binary_masks_default = burn_binary_masks_env in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(
        description=(
            "Build semantic segmentation datasets from HOT Tasking Manager OAM projects, "
            "labelled with any OpenStreetMap feature type"
        )
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
        "--generate-parquet",
        action="store_true",
        help="Generate HuggingFace-compatible Parquet from existing dataset",
    )
    parser.add_argument(
        "--generate-splits",
        action="store_true",
        help=(
            "Generate train/validation/test split assignments based on project clusters. "
            "Projects sharing the same imagery URL are kept in the same split to prevent "
            "spatial leakage. Writes splits.json to the output directory"
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
        "--label-buffer-meters",
        type=float,
        default=0.0,
        help=(
            "Widen label geometries by this many metres before burning masks. "
            "Line features need a buffer to produce usable masks (default: 0)"
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
    return parser.parse_args(argv)


def _run_standalone_command(args, output_dir: Path) -> bool:
    if args.generate_splits:
        generate_splits(output_dir)
        return True

    if args.generate_parquet:
        generate_parquet(output_dir, dataset_version=args.dataset_version)
        return True

    if args.push_to_hub is not None:
        _require_repo_id(args.push_to_hub, "--push-to-hub")
        push_to_hub(output_dir, args.push_to_hub, dataset_version=args.dataset_version)
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

    spec = resolve_feature_spec(args.feature_type, args.tm_mapping_type, args.geometry_types)
    logger.info(
        "Feature type '%s' -> TM mapping type %s, geometry %s",
        spec.osm_tag,
        spec.tm_mapping_type,
        "/".join(spec.geometry_types),
    )

    projects = _collect_projects(args, spec, output_dir)

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
    )
    failed_project_ids = build_all_projects(projects, config, dataset_version=args.dataset_version)
    if failed_project_ids:
        sys.exit(1)


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
