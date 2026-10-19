set dotenv-load

default_output := "dataset"

# List available recipes
default:
    @just --list

# Install deps and pre-commit hooks
setup:
    uv sync --all-groups
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# Run all checks: ruff fix + format, ty typecheck, pre-commit hooks, pytest
lint:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/
    uv run ty check src/
    uv run pre-commit run --all-files
    uv run pytest tests/ -v

# Run tests only
test:
    uv run pytest tests/ -v

# Build dataset. Pass extra CLI flags as needed (e.g. just build --limit 0 --years-back 3)
build *args:
    uv run hot-oam-dataset --output-dir {{ default_output }} {{ args }}
    uv run hot-oam-dataset --compress-raw --output-dir {{ default_output }}

# Build a hand-picked set of TM projects into a named folder. Pass --feature-type and friends via args
build-projects ids output_dir *args:
    uv run hot-oam-dataset --project-ids {{ ids }} --output-dir {{ output_dir }} {{ args }}
    uv run hot-oam-dataset --compress-raw --output-dir {{ output_dir }}

# Compress raw chips/masks/labels into per-project tar.gz archives
compress-raw output_dir=default_output:
    uv run hot-oam-dataset --compress-raw --output-dir {{ output_dir }}

# Publish to HuggingFace Hub. Add "raw" as a third arg to also upload raw files (e.g. just publish REPO dataset raw)
publish repo_id output_dir=default_output raw="":
    uv run hot-oam-dataset --generate-parquet --output-dir {{ output_dir }}
    uv run hot-oam-dataset --push-to-hub {{ repo_id }} --output-dir {{ output_dir }}
    {{ if raw =~ "raw" { "uv run hot-oam-dataset --push-raw " + repo_id + " --compress-raw --output-dir " + output_dir } else { "@echo 'Done (skipped raw upload)'" } }}

# Publish a named hand-picked dataset folder. Add "raw" as a third arg to also upload raw files.
publish-projects output_dir repo_id raw="":
    just publish {{ repo_id }} {{ output_dir }} {{ raw }}

# Build one dataset from its YAML config: tiles, splits, Parquet and STAC
dataset config:
    uv run hot-oam-dataset --config {{ config }} --all
