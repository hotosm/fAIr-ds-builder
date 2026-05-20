import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

TM_API_BASE = "https://tasking-manager-production-api.hotosm.org"
PROJECTS_ENDPOINT = "/api/v2/projects/"
OAM_DOMAIN = "openaerialmap.org"
MINIMUM_VALIDATION_PERCENT = 95
PROJECT_STATUSES = ("PUBLISHED", "ARCHIVED")


@dataclass
class TaskingManagerProject:
    project_id: int
    name: str
    status: str
    percent_mapped: int
    percent_validated: int
    imagery_url: str
    area_of_interest: dict
    bbox: list[float]
    mapping_types: list[str]
    organisation: str
    created: str
    country: str = ""


@dataclass
class TaskingManagerClient:
    base_url: str = TM_API_BASE
    timeout: int = 30
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def iter_custom_imagery_project_ids(
        self,
        created_from: str | None = None,
        created_to: str | None = None,
        mapping_types: str = "BUILDINGS",
    ) -> Generator[int, None, None]:
        for status in PROJECT_STATUSES:
            page = 1
            while True:
                params = {
                    "imagery": "custom",
                    "omitMapResults": "true",
                    "mappingTypes": mapping_types,
                    "projectStatuses": status,
                    "page": str(page),
                }
                if created_from:
                    params["createdFrom"] = created_from
                if created_to:
                    params["createdTo"] = created_to

                logger.info("Fetching %s projects page %d", status, page)
                response = self._client.get(PROJECTS_ENDPOINT, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for project in results:
                    yield project["projectId"]

                total_pages = data.get("pagination", {}).get("pages", 1)
                if page >= total_pages:
                    break
                page += 1

    def fetch_project_summary(self, project_id: int) -> dict:
        endpoint = f"{PROJECTS_ENDPOINT}{project_id}/queries/summary/"
        response = self._client.get(endpoint)
        response.raise_for_status()
        return response.json()

    def fetch_project_full(self, project_id: int) -> dict:
        response = self._client.get(f"{PROJECTS_ENDPOINT}{project_id}/")
        response.raise_for_status()
        return response.json()

    def is_oam_project(self, project_detail: dict) -> bool:
        imagery = project_detail.get("imagery", "") or ""
        return OAM_DOMAIN in imagery

    def is_validated_building_project(self, project_detail: dict) -> bool:
        percent_validated = project_detail.get("percentValidated", 0)
        mapping_types = project_detail.get("mappingTypes", [])
        return percent_validated >= MINIMUM_VALIDATION_PERCENT and "BUILDINGS" in mapping_types

    def build_project_record(self, detail: dict) -> TaskingManagerProject:
        country_tags = detail.get("countryTag", [])
        return TaskingManagerProject(
            project_id=detail["projectId"],
            name=detail.get("projectInfo", {}).get("name", ""),
            status=detail.get("status", ""),
            percent_mapped=detail.get("percentMapped", 0),
            percent_validated=detail.get("percentValidated", 0),
            imagery_url=detail.get("imagery", ""),
            area_of_interest=detail.get("areaOfInterest", {}),
            bbox=detail.get("aoiBBOX", []),
            mapping_types=detail.get("mappingTypes", []),
            organisation=detail.get("organisationName", ""),
            created=detail.get("created", ""),
            country=country_tags[0] if country_tags else "",
        )


def fetch_projects_by_ids(
    project_ids: list[int],
    output_path: Path | None = None,
) -> list[TaskingManagerProject]:
    """Fetch a hand-picked set of TM projects by ID.

    Skips the discovery search and validation gate. Warns when a project falls
    below the standard validation threshold or does not use OAM imagery, but
    builds the record anyway so the caller can proceed.
    """
    projects: list[TaskingManagerProject] = []

    with TaskingManagerClient() as client:
        for pid in project_ids:
            detail = client.fetch_project_full(pid)

            if not client.is_validated_building_project(detail):
                logger.warning(
                    "Project %d is below %d%% validation or not a buildings project; "
                    "proceeding because it was hand-picked",
                    pid,
                    MINIMUM_VALIDATION_PERCENT,
                )

            if not client.is_oam_project(detail):
                logger.warning(
                    "Project %d does not use OAM imagery; proceeding because it was "
                    "hand-picked (imagery URL: %s)",
                    pid,
                    detail.get("imagery", ""),
                )

            project = client.build_project_record(detail)
            projects.append(project)
            logger.info(
                "Loaded hand-picked #%d %s (%s)",
                project.project_id,
                project.name,
                project.country,
            )

        logger.info("Hand-picked projects loaded: %d", len(projects))

        if output_path:
            save_discovery_outputs(projects, output_path)

        return projects


def discover_oam_building_projects(
    created_from: str | None = None,
    created_to: str | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
) -> list[TaskingManagerProject]:
    with TaskingManagerClient() as client:
        project_id_iter = client.iter_custom_imagery_project_ids(
            created_from=created_from,
            created_to=created_to,
        )

        projects = []
        for pid in project_id_iter:
            if limit and len(projects) >= limit:
                break

            try:
                summary = client.fetch_project_summary(pid)
            except httpx.HTTPStatusError as exc:
                logger.warning("Failed to fetch summary for project %d: %s", pid, exc)
                continue

            if not client.is_validated_building_project(summary):
                logger.debug("Project %d not validated for buildings, skipping", pid)
                continue

            try:
                detail = client.fetch_project_full(pid)
            except httpx.HTTPStatusError as exc:
                logger.warning("Failed to fetch detail for project %d: %s", pid, exc)
                continue

            if not client.is_oam_project(detail):
                logger.debug("Project %d not using OAM imagery, skipping", pid)
                continue

            project = client.build_project_record(detail)
            projects.append(project)
            logger.info(
                "Discovered #%d %s (%s)",
                project.project_id,
                project.name,
                project.country,
            )

        logger.info("Total validated OAM building projects: %d", len(projects))

        if output_path:
            save_discovery_outputs(projects, output_path)

        return projects


def save_discovery_outputs(projects: list[TaskingManagerProject], output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)

    summary = [
        {
            "project_id": p.project_id,
            "name": p.name,
            "status": p.status,
            "percent_validated": p.percent_validated,
            "imagery_url": p.imagery_url,
            "organisation": p.organisation,
            "country": p.country,
            "created": p.created,
            "mapping_types": p.mapping_types,
        }
        for p in projects
    ]
    (output_path / "projects_summary.json").write_text(json.dumps(summary, indent=2))

    geojson_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "project_id": p.project_id,
                    "name": p.name,
                    "imagery_url": p.imagery_url,
                    "percent_validated": p.percent_validated,
                    "organisation": p.organisation,
                    "country": p.country,
                },
                "geometry": p.area_of_interest,
            }
            for p in projects
        ],
    }
    (output_path / "projects_map.geojson").write_text(json.dumps(geojson_collection, indent=2))

    logger.info("Saved discovery outputs to %s", output_path)
