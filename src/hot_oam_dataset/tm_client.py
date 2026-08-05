import json
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from hot_oam_dataset.feature_spec import DEFAULT_FEATURE_TYPE, KNOWN_FEATURE_SPECS

logger = logging.getLogger(__name__)

TM_API_BASE = "https://tasking-manager-production-api.hotosm.org"
PROJECTS_ENDPOINT = "/api/v2/projects/"
OAM_DOMAIN = "openaerialmap.org"
MINIMUM_VALIDATION_PERCENT = 95
PROJECT_STATUSES = ("PUBLISHED", "ARCHIVED")
DEFAULT_MAPPING_TYPE = KNOWN_FEATURE_SPECS[DEFAULT_FEATURE_TYPE].tm_mapping_type


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
        mapping_type: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> Generator[int, None, None]:
        """Yield custom-imagery project ids, across every mapping type when none is given."""
        for status in PROJECT_STATUSES:
            page = 1
            while True:
                params = {
                    "imagery": "custom",
                    "omitMapResults": "true",
                    "projectStatuses": status,
                    "page": str(page),
                }
                if mapping_type:
                    params["mappingTypes"] = mapping_type
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

    def is_validated_project(self, project_detail: dict, mapping_type: str) -> bool:
        percent_validated = project_detail.get("percentValidated", 0)
        mapping_types = project_detail.get("mappingTypes", [])
        return percent_validated >= MINIMUM_VALIDATION_PERCENT and mapping_type in mapping_types

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
    mapping_type: str = DEFAULT_MAPPING_TYPE,
    output_path: Path | None = None,
) -> list[TaskingManagerProject]:
    """Fetch TM projects by ID, warning rather than skipping on low validation or non-OAM."""
    projects: list[TaskingManagerProject] = []

    with TaskingManagerClient() as client:
        for pid in project_ids:
            detail = client.fetch_project_full(pid)

            if not client.is_validated_project(detail, mapping_type):
                logger.warning(
                    "Project %d is below %d%% validation or not a %s project; "
                    "proceeding because it was hand-picked",
                    pid,
                    MINIMUM_VALIDATION_PERCENT,
                    mapping_type,
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


def discover_oam_projects(
    mapping_type: str = DEFAULT_MAPPING_TYPE,
    created_from: str | None = None,
    created_to: str | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
) -> list[TaskingManagerProject]:
    with TaskingManagerClient() as client:
        project_id_iter = client.iter_custom_imagery_project_ids(
            mapping_type=mapping_type,
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

            if not client.is_validated_project(summary, mapping_type):
                logger.debug("Project %d not validated for %s, skipping", pid, mapping_type)
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

        logger.info("Total validated OAM %s projects: %d", mapping_type, len(projects))

        if output_path:
            save_discovery_outputs(projects, output_path)

        return projects


def _summary_record(project: TaskingManagerProject) -> dict:
    return {
        "project_id": project.project_id,
        "name": project.name,
        "status": project.status,
        "percent_validated": project.percent_validated,
        "imagery_url": project.imagery_url,
        "organisation": project.organisation,
        "country": project.country,
        "created": project.created,
        "mapping_types": project.mapping_types,
    }


def _map_feature(project: TaskingManagerProject) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "project_id": project.project_id,
            "name": project.name,
            "imagery_url": project.imagery_url,
            "percent_validated": project.percent_validated,
            "organisation": project.organisation,
            "country": project.country,
        },
        "geometry": project.area_of_interest,
    }


def _merge_by_project_id(
    existing: list[dict],
    incoming: list[dict],
    project_id_of: Callable[[dict], int],
) -> list[dict]:
    merged = {project_id_of(record): record for record in existing}
    for record in incoming:
        merged[project_id_of(record)] = record
    return list(merged.values())


def _read_existing_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _read_existing_features(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("features", []) if path.exists() else []


def save_discovery_outputs(projects: list[TaskingManagerProject], output_path: Path):
    """Write discovery outputs, keeping projects already recorded under output_path."""
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "projects_summary.json"
    map_path = output_path / "projects_map.geojson"

    summary = _merge_by_project_id(
        _read_existing_summary(summary_path),
        [_summary_record(p) for p in projects],
        lambda record: record["project_id"],
    )
    features = _merge_by_project_id(
        _read_existing_features(map_path),
        [_map_feature(p) for p in projects],
        lambda feature: feature["properties"]["project_id"],
    )

    summary_path.write_text(json.dumps(summary, indent=2))
    map_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))

    logger.info("Saved discovery outputs for %d projects to %s", len(summary), output_path)
