default_output := "dataset"

# List available recipes
default:
    @just --list

# Run ruff linter
lint:
    uv run ruff check src/ tests/

# Run ruff formatter check
format-check:
    uv run ruff format --check src/ tests/

# Auto-format code with ruff
format:
    uv run ruff format src/ tests/

# Run ty type checker
typecheck:
    uv run ty check src/

# Run pytest
test:
    uv run pytest tests/ -v

# Run all checks (lint + format + typecheck + test)
check: lint format-check typecheck test

# Discover validated OAM building projects without downloading
discover output=default_output:
    uv run hot-oam-dataset --discover-only --output-dir {{ output }} --verbose

# Build dataset with default limit (5 projects)
build output=default_output:
    uv run hot-oam-dataset --output-dir {{ output }} --verbose

# Build full dataset (all projects)
build-all output=default_output:
    uv run hot-oam-dataset --limit 0 --output-dir {{ output }} --verbose

# Build dataset with custom limit and time range
build-custom limit years_back output=default_output:
    uv run hot-oam-dataset --limit {{ limit }} --years-back {{ years_back }} --output-dir {{ output }} --verbose

# Generate HuggingFace Parquet from existing dataset
parquet output=default_output:
    uv run hot-oam-dataset --generate-parquet --output-dir {{ output }}

# Push Parquet dataset to HuggingFace Hub
push-hub repo_id output=default_output:
    uv run hot-oam-dataset --push-to-hub {{ repo_id }} --output-dir {{ output }}

# Upload raw chips/labels/metadata to HuggingFace Hub
push-raw repo_id output=default_output:
    uv run hot-oam-dataset --push-raw {{ repo_id }} --output-dir {{ output }}

# Full publish: generate Parquet, push dataset and raw files to HuggingFace Hub
publish repo_id output=default_output: (parquet output) (push-hub repo_id output) (push-raw repo_id output)
    @echo "Published to https://huggingface.co/datasets/{{ repo_id }}"
