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

# Compress raw chips/masks/labels into per-project tar.gz archives
compress-raw:
    uv run hot-oam-dataset --compress-raw --output-dir {{ default_output }}

# Publish to HuggingFace Hub. Add "raw" to also upload raw files (e.g. just publish REPO raw)
publish repo_id raw="":
    uv run hot-oam-dataset --generate-parquet --output-dir {{ default_output }}
    uv run hot-oam-dataset --push-to-hub {{ repo_id }} --output-dir {{ default_output }}
    {{ if raw =~ "raw" { "uv run hot-oam-dataset --push-raw " + repo_id + " --compress-raw --output-dir " + default_output } else { "@echo 'Done (skipped raw upload)'" } }}
