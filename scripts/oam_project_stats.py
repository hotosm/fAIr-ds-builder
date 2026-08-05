"""Summarise every OAM project on the Tasking Manager and dump them to oam_projects.csv."""

import logging

import httpx
import pandas as pd

from hot_oam_dataset.tm_client import MINIMUM_VALIDATION_PERCENT, TaskingManagerClient

CSV_PATH = "oam_projects.csv"
VALIDATION_BUCKETS = (100, MINIMUM_VALIDATION_PERCENT, 80)


def build_project_row(client: TaskingManagerClient, detail: dict) -> dict:
    project = client.build_project_record(detail)
    return {
        "project_id": project.project_id,
        "name": project.name,
        "status": project.status,
        "percent_mapped": project.percent_mapped,
        "percent_validated": project.percent_validated,
        "is_oam": client.is_oam_project(detail),
        "has_buildings": "BUILDINGS" in project.mapping_types,
        "mapping_types": project.mapping_types,
        "organisation": project.organisation,
        "country": project.country,
        "created": project.created,
        "imagery_url": project.imagery_url,
    }


def collect_rows(client: TaskingManagerClient) -> tuple[list[dict], int]:
    rows = []
    skipped = 0

    for index, project_id in enumerate(client.iter_custom_imagery_project_ids(), 1):
        try:
            detail = client.fetch_project_full(project_id)
        except httpx.HTTPError as exc:
            print(f"  Skipping project {project_id}: {exc}")
            skipped += 1
            continue

        rows.append(build_project_row(client, detail))

        if index % 25 == 0:
            print(f"  Fetched {index} project details...")

    return rows, skipped


def print_summary(frame: pd.DataFrame, oam: pd.DataFrame, skipped: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"Skipped (detail fetch failed): {skipped}")
    print(f"Total custom imagery projects: {len(frame)}")
    print(f"OAM projects: {len(oam)}")

    print("\n--- OAM project breakdown ---")
    print(f"With BUILDINGS mapping type: {oam['has_buildings'].sum()}")
    print(f"Without BUILDINGS type:      {(~oam['has_buildings']).sum()}")

    print("\n--- Validation (OAM only) ---")
    for threshold in VALIDATION_BUCKETS:
        print(f">={threshold}% validated: {(oam['percent_validated'] >= threshold).sum()}")

    qualified = oam[oam["has_buildings"] & (oam["percent_validated"] >= MINIMUM_VALIDATION_PERCENT)]
    print(f"\nOAM + BUILDINGS + >={MINIMUM_VALIDATION_PERCENT}% validated: {len(qualified)}")

    print("\n--- Status distribution (OAM) ---")
    print(oam["status"].value_counts().to_string())
    print("\n--- Top organisations (OAM) ---")
    print(oam["organisation"].value_counts().head(10).to_string())
    print("\n--- Top countries (OAM) ---")
    print(oam["country"].value_counts().head(15).to_string())


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with TaskingManagerClient() as client:
        print("Fetching custom imagery projects...")
        rows, skipped = collect_rows(client)

    frame = pd.DataFrame(rows)
    oam = frame[frame["is_oam"]]

    print_summary(frame, oam, skipped)

    oam.to_csv(CSV_PATH, index=False)
    print(f"\nOAM projects saved to {CSV_PATH} (shape {oam.shape})")


if __name__ == "__main__":
    main()
