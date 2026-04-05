import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from hot_oam_dataset import __version__
from hot_oam_dataset.dataset_builder import DatasetConfig, build_all_projects
from hot_oam_dataset.hf_publisher import (
    compress_projects,
    generate_parquet,
    push_raw_to_hub,
    push_to_hub,
)
from hot_oam_dataset.tm_client import discover_oam_building_projects


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    hf_repo_id_from_env = os.getenv("HF_DATASET_REPO_ID", "")
    feature_type_from_env = os.getenv("HOT_OAM_FEATURE_TYPE", "building")
    burn_binary_masks_env = os.getenv("HOT_OAM_BURN_BINARY_MASKS", "true").strip().lower()
    burn_binary_masks_default = burn_binary_masks_env in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(
        description=(
            "Build semantic segmentation datasets for building disaster mapping "
            "from HOT Tasking Manager OAM projects"
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
            "OSM feature type to fetch (default: HOT_OAM_FEATURE_TYPE env or building). "
            "Examples: building, highway"
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


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_parquet:
        generate_parquet(output_dir, dataset_version=args.dataset_version)
        return

    if args.push_to_hub:
        push_to_hub(output_dir, args.push_to_hub, dataset_version=args.dataset_version)
        return

    if args.push_to_hub is not None and not args.push_to_hub:
        logging.error("REPO_ID missing. Pass --push-to-hub <repo_id> or set HF_DATASET_REPO_ID")
        sys.exit(2)

    if args.compress_raw and not args.push_raw:
        compress_projects(output_dir)
        return

    if args.push_raw:
        push_raw_to_hub(
            output_dir,
            args.push_raw,
            dataset_version=args.dataset_version,
            compress=args.compress_raw,
        )
        return

    if args.push_raw is not None and not args.push_raw:
        logging.error("REPO_ID missing. Pass --push-raw <repo_id> or set HF_DATASET_REPO_ID")
        sys.exit(2)

    now = datetime.now(tz=UTC)
    created_from = now.replace(year=now.year - args.years_back).strftime("%Y-%m-%d")
    created_to = now.strftime("%Y-%m-%d")

    logging.info("Searching TM projects from %s to %s", created_from, created_to)

    effective_limit = args.limit or None

    projects = discover_oam_building_projects(
        created_from=created_from,
        created_to=created_to,
        output_path=output_dir,
        limit=effective_limit,
    )

    if not projects:
        logging.warning("No validated OAM building projects found")
        sys.exit(0)

    _print_projects_table(projects)

    if args.discover_only:
        logging.info("Discovery complete. Saved to %s", output_dir)
        return

    config = DatasetConfig(
        output_dir=output_dir,
        zoom=args.zoom,
        feature_type=args.feature_type,
        burn_binary_masks=args.burn_binary_masks,
    )
    build_all_projects(projects, config, dataset_version=args.dataset_version)


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
