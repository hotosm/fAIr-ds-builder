"""Fetch all OAM projects from TM API and produce a summary DataFrame."""

import httpx
import pandas as pd

TM_API = "https://tasking-manager-production-api.hotosm.org"
PROJECTS_EP = "/api/v2/projects/"
OAM_DOMAIN = "openaerialmap.org"
STATUSES = ("PUBLISHED", "ARCHIVED")


def fetch_custom_imagery_project_ids(client: httpx.Client) -> list[int]:
    ids = []
    for status in STATUSES:
        page = 1
        while True:
            params = {
                "imagery": "custom",
                "omitMapResults": "true",
                "projectStatuses": status,
                "page": str(page),
            }
            resp = client.get(PROJECTS_EP, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            ids.extend(p["projectId"] for p in results)
            total_pages = data.get("pagination", {}).get("pages", 1)
            print(f"  {status} page {page}/{total_pages} ({len(results)} projects)")
            if page >= total_pages:
                break
            page += 1
    return ids


def fetch_project_detail(client: httpx.Client, pid: int) -> dict | None:
    try:
        resp = client.get(f"{PROJECTS_EP}{pid}/")
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError:
        return None


def main():
    with httpx.Client(base_url=TM_API, timeout=30) as client:
        print("Fetching custom imagery project IDs...")
        all_ids = fetch_custom_imagery_project_ids(client)
        print(f"\nTotal custom imagery projects: {len(all_ids)}")

        rows = []
        for i, pid in enumerate(all_ids, 1):
            detail = fetch_project_detail(client, pid)
            if not detail:
                continue

            imagery = detail.get("imagery", "") or ""
            is_oam = OAM_DOMAIN in imagery
            mapping_types = detail.get("mappingTypes", [])
            country_tags = detail.get("countryTag", [])

            rows.append(
                {
                    "project_id": pid,
                    "name": detail.get("projectInfo", {}).get("name", ""),
                    "status": detail.get("status", ""),
                    "percent_mapped": detail.get("percentMapped", 0),
                    "percent_validated": detail.get("percentValidated", 0),
                    "is_oam": is_oam,
                    "has_buildings": "BUILDINGS" in mapping_types,
                    "mapping_types": mapping_types,
                    "organisation": detail.get("organisationName", ""),
                    "country": country_tags[0] if country_tags else "",
                    "created": detail.get("created", ""),
                    "imagery_url": imagery,
                }
            )

            if i % 25 == 0:
                print(f"  Fetched {i}/{len(all_ids)} details...")

        df = pd.DataFrame(rows)
        oam = df[df["is_oam"]]

        print(f"\n{'=' * 60}")
        print(f"Total custom imagery projects: {len(df)}")
        print(f"OAM projects: {len(oam)}")
        print("\n--- OAM project breakdown ---")
        print(f"With BUILDINGS mapping type: {oam['has_buildings'].sum()}")
        print(f"Without BUILDINGS type:      {(~oam['has_buildings']).sum()}")
        print("\n--- Validation (OAM only) ---")
        print(f"100% validated: {(oam['percent_validated'] == 100).sum()}")
        print(f">=95% validated: {(oam['percent_validated'] >= 95).sum()}")
        print(f">=80% validated: {(oam['percent_validated'] >= 80).sum()}")
        print(f"<80% validated:  {(oam['percent_validated'] < 80).sum()}")
        print("\n--- OAM + BUILDINGS + >=95% validated ---")
        qualified = oam[(oam["has_buildings"]) & (oam["percent_validated"] >= 95)]
        print(f"Qualified projects: {len(qualified)}")
        print("\n--- Status distribution (OAM) ---")
        print(oam["status"].value_counts().to_string())
        print("\n--- Top organisations (OAM) ---")
        print(oam["organisation"].value_counts().head(10).to_string())
        print("\n--- Top countries (OAM) ---")
        print(oam["country"].value_counts().head(15).to_string())

        csv_path = "oam_projects.csv"
        oam.to_csv(csv_path, index=False)
        print(f"\nOAM DataFrame saved to {csv_path}")
        print(f"Full DataFrame shape: {df.shape}, OAM shape: {oam.shape}")


if __name__ == "__main__":
    main()
