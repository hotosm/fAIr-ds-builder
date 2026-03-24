default_output := "dataset"

# List available recipes
default:
    @just --list

# Run all checks: ruff lint, ruff format, ty typecheck, pytest
lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run ty check src/
    uv run pytest tests/ -v

# Run tests only
test:
    uv run pytest tests/ -v

# Build dataset. Pass extra CLI flags as needed (e.g. just build --limit 0 --years-back 3)
build *args:
    uv run hot-oam-dataset --output-dir {{ default_output }} --verbose {{ args }}

# Publish to HuggingFace Hub. Add "raw" to also upload raw files (e.g. just publish REPO raw)
publish repo_id raw="":
    uv run hot-oam-dataset --generate-parquet --output-dir {{ default_output }}
    uv run hot-oam-dataset --push-to-hub {{ repo_id }} --output-dir {{ default_output }}
    {{ if raw =~ "raw" { "uv run hot-oam-dataset --push-raw " + repo_id + " --output-dir " + default_output } else { "@echo 'Done (skipped raw upload)'" } }}
