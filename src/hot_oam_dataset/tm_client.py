import json
import logging
import os
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from shapely import union_all
from shapely.geometry import mapping, shape

from hot_oam_dataset.feature_spec import DEFAULT_FEATURE_TYPE, KNOWN_FEATURE_SPECS

logger = logging.getLogger(__name__)

TM_API_BASE = "https://tasking-manager-production-api.hotosm.org"
PROJECTS_ENDPOINT = "/api/v2/projects/"
OAM_IMAGERY_MARKERS = (
    "openaerialmap.org",
    "api.imagery.hotosm.org/raster/collections/openaerialmap/",
)
TM_TOKEN_ENV = "TM_TOKEN"
TOKEN_SCHEME = "Token"
VALIDATED_TASK_STATUS = "VALIDATED"
MINIMUM_VALIDATION_PERCENT = 95
PROJECT_STATUSES = ("PUBLISHED", "ARCHIVED")
DEFAULT_MAPPING_TYPE = KNOWN_FEATURE_SPECS[DEFAULT_FEATURE_TYPE].tm_mapping_type
IMAGERY_PREFIX_PATTERN = re.compile(r"^[a-zA-Z]+\s*(\[\d+\])?:(?=https?://)")


def normalize_imagery_url(imagery: str) -> str:
    """Strip the editor prefix TM stores on some layers, such as tms[22]:https://..."""
    return IMAGERY_PREFIX_PATTERN.sub("", imagery or "")


def read_tm_token() -> str | None:
    """Read the TM session token, accepting it with or without the leading 'Token ' scheme."""
    value = os.getenv(TM_TOKEN_ENV, "").strip()
    return value.removeprefix(f"{TOKEN_SCHEME} ").strip() or None


def validated_task_area(detail: dict) -> tuple[dict, int, int]:
    """Union VALIDATED task squares clipped to the AOI, since squares extend past it."""
    tasks = detail["tasks"]["features"]
    validated = [
        task for task in tasks if task["properties"].get("taskStatus") == VALIDATED_TASK_STATUS
    ]
    if not validated:
        return {}, 0, len(tasks)
    union = union_all([shape(task["geometry"]) for task in validated])
    clipped = union.intersection(shape(detail["areaOfInterest"]))
    if clipped.area == 0:
        return {}, 0, len(tasks)
    return mapping(clipped), len(validated), len(tasks)


class PrivateProjectError(PermissionError):
    """Raised when TM refuses a project because it is private to the caller."""


@dataclass
class TaskingManagerProject:
    project_id: int
    name: str
    status: str
    percent_mapped: int
    percent_validated: int
    imagery_url: str
    area_of_interest: dict
    validated_area: dict
    validated_task_count: int
    total_task_count: int
    bbox: list[float]
    mapping_types: list[str]
    organisation: str
    created: str
    country: str = ""


@dataclass
class TaskingManagerClient:
    base_url: str = TM_API_BASE
    timeout: int = 30
    token: str | None = field(default_factory=read_tm_token, repr=False)
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self):
        headers = {"Authorization": f"{TOKEN_SCHEME} {self.token}"} if self.token else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=headers)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def iter_projects(
        self,
        mapping_type: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> Generator[dict, None, None]:
        """Yield project list entries, across every mapping type when none is given."""
        for status in PROJECT_STATUSES:
            page = 1
            while True:
                params = {
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

                yield from results

                total_pages = data.get("pagination", {}).get("pages", 1)
                if page >= total_pages:
                    break
                page += 1

    def fetch_project_summary(self, project_id: int) -> dict:
        endpoint = f"{PROJECTS_ENDPOINT}{project_id}/queries/summary/"
        return self._get_project_json(project_id, endpoint)

    def fetch_project_full(self, project_id: int) -> dict:
        return self._get_project_json(project_id, f"{PROJECTS_ENDPOINT}{project_id}/")

    def _get_project_json(self, project_id: int, endpoint: str) -> dict:
        response = self._client.get(endpoint)
        if response.status_code == httpx.codes.FORBIDDEN:
            raise PrivateProjectError(
                f"TM refused project {project_id} (403). If it is private, set {TM_TOKEN_ENV} "
                f"to a TM session token with access to it ({response.text})"
            )
        response.raise_for_status()
        return response.json()

    def is_oam_project(self, project_detail: dict) -> bool:
        imagery = project_detail.get("imagery", "") or ""
        return any(marker in imagery for marker in OAM_IMAGERY_MARKERS)

    def is_validated_project(self, project_detail: dict, mapping_type: str) -> bool:
        percent_validated = project_detail.get("percentValidated", 0)
        mapping_types = project_detail.get("mappingTypes", [])
        return percent_validated >= MINIMUM_VALIDATION_PERCENT and mapping_type in mapping_types

    def build_project_record(self, detail: dict) -> TaskingManagerProject:
        country_tags = detail.get("countryTag", [])
        project_info = detail.get("projectInfo") or {}
        validated_area, validated_task_count, total_task_count = validated_task_area(detail)
        return TaskingManagerProject(
            project_id=detail["projectId"],
            name=project_info.get("name", "").strip(),
            status=detail.get("status", ""),
            percent_mapped=detail.get("percentMapped", 0),
            percent_validated=detail.get("percentValidated", 0),
            imagery_url=normalize_imagery_url(detail.get("imagery", "")),
            area_of_interest=detail.get("areaOfInterest", {}),
            validated_area=validated_area,
            validated_task_count=validated_task_count,
            total_task_count=total_task_count,
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
            if not project.validated_task_count:
                logger.warning("Project %d has no validated tasks, skipping", pid)
                continue
            projects.append(project)
            logger.info(
                "Loaded hand-picked #%d %s (%s), %d of %d tasks validated",
                project.project_id,
                project.name,
                project.country,
                project.validated_task_count,
                project.total_task_count,
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
    """Discover validated OAM projects, narrowing on the cheapest field available first."""
    projects: list[TaskingManagerProject] = []
    seen = under_validated = not_oam = 0

    with TaskingManagerClient() as client:
        for entry in client.iter_projects(
            mapping_type=mapping_type,
            created_from=created_from,
            created_to=created_to,
        ):
            if limit and len(projects) >= limit:
                break

            seen += 1
            pid = entry["projectId"]

            if entry.get("percentValidated", 0) < MINIMUM_VALIDATION_PERCENT:
                under_validated += 1
                continue

            try:
                summary = client.fetch_project_summary(pid)
            except (httpx.HTTPStatusError, PrivateProjectError) as exc:
                logger.warning("Failed to fetch summary for project %d: %s", pid, exc)
                continue

            if not client.is_oam_project(summary):
                not_oam += 1
                continue

            if not client.is_validated_project(summary, mapping_type):
                logger.debug("Project %d not validated for %s, skipping", pid, mapping_type)
                continue

            try:
                detail = client.fetch_project_full(pid)
            except (httpx.HTTPStatusError, PrivateProjectError) as exc:
                logger.warning("Failed to fetch detail for project %d: %s", pid, exc)
                continue

            project = client.build_project_record(detail)
            if not project.validated_task_count:
                logger.warning("Project %d has no validated tasks, skipping", pid)
                continue
            projects.append(project)
            logger.info(
                "Discovered #%d %s (%s)",
                project.project_id,
                project.name,
                project.country,
            )

        logger.info(
            "Scanned %d %s projects: %d below %d%% validation, %d not OAM, %d kept",
            seen,
            mapping_type,
            under_validated,
            MINIMUM_VALIDATION_PERCENT,
            not_oam,
            len(projects),
        )

        if output_path:
            save_discovery_outputs(projects, output_path)

        return projects


def _summary_record(project: TaskingManagerProject) -> dict:
    return {
        "project_id": project.project_id,
        "name": project.name,
        "status": project.status,
        "percent_validated": project.percent_validated,
        "validated_task_count": project.validated_task_count,
        "total_task_count": project.total_task_count,
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
