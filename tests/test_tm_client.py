import json
import logging

import httpx
import pytest
from shapely.geometry import Point, box, mapping, shape

from hot_oam_dataset.tm_client import (
    PrivateProjectError,
    TaskingManagerClient,
    TaskingManagerProject,
    discover_oam_projects,
    fetch_projects_by_ids,
    read_tm_token,
    save_discovery_outputs,
    validated_task_area,
)


def _page(project_ids: list[int], total_pages: int) -> dict:
    return {
        "results": [{"projectId": pid} for pid in project_ids],
        "pagination": {"pages": total_pages},
    }


def _summary(percent_validated: int, mapping_types: list[str]) -> dict:
    return {"percentValidated": percent_validated, "mappingTypes": mapping_types}


def _task(min_x: float, status: str) -> dict:
    ring = [[min_x, 0.0], [min_x + 1, 0.0], [min_x + 1, 1.0], [min_x, 1.0], [min_x, 0.0]]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"taskStatus": status},
    }


def _detail(imagery: str, project_id: int, task_statuses: tuple[str, ...] = ("VALIDATED",)) -> dict:
    return {
        "tasks": {
            "type": "FeatureCollection",
            "features": [_task(float(i), status) for i, status in enumerate(task_statuses)],
        },
        "areaOfInterest": mapping(box(0.0, 0.0, float(len(task_statuses)), 1.0)),
        "projectId": project_id,
        "imagery": imagery,
        "projectInfo": {"name": f"Project {project_id}"},
        "status": "PUBLISHED",
        "percentMapped": 100,
        "percentValidated": 100,
        "mappingTypes": ["BUILDINGS"],
        "aoiBBOX": [0.0, 0.0, 1.0, 1.0],
        "organisationName": "HOT",
        "created": "2025-01-01T00:00:00Z",
        "countryTag": ["Nepal"],
    }


def _make_client(mocker, pages: list[dict]) -> TaskingManagerClient:
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    http = mocker.MagicMock()
    http.get.side_effect = [mocker.MagicMock(**{"json.return_value": p}) for p in pages]
    client._client = http  # noqa: SLF001
    return client


def test_iter_ids_yields_all_pages(mocker):
    client = _make_client(
        mocker,
        [_page([1, 2, 3], total_pages=2), _page([4, 5], total_pages=2), _page([], total_pages=1)],
    )
    ids = [p["projectId"] for p in client.iter_projects("BUILDINGS")]
    assert ids == [1, 2, 3, 4, 5]


def test_iter_ids_generator_is_lazy(mocker):
    client = _make_client(mocker, [_page(list(range(1, 11)), total_pages=5)])

    gen = client.iter_projects("BUILDINGS")
    first_three = [next(gen)["projectId"] for _ in range(3)]

    assert first_three == [1, 2, 3]
    assert client._client.get.call_count == 1  # noqa: SLF001


def test_is_oam_project_true():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert client.is_oam_project({"imagery": "https://tiles.openaerialmap.org/abc"})


def test_is_oam_project_false():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_oam_project({"imagery": "https://example.com/tiles"})


def test_is_oam_project_missing_imagery():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_oam_project({})


def test_is_validated_project_passes():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert client.is_validated_project(_summary(95, ["BUILDINGS"]), "BUILDINGS")


def test_is_validated_project_below_threshold():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_validated_project(_summary(94, ["BUILDINGS"]), "BUILDINGS")


def test_is_validated_project_wrong_type():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_validated_project(_summary(100, ["ROADS"]), "BUILDINGS")


def _make_hand_picked_client(mocker, details_by_id: dict[int, dict]):
    real_client = TaskingManagerClient.__new__(TaskingManagerClient)
    mock_instance = mocker.MagicMock()
    mock_instance.fetch_project_full.side_effect = lambda pid: details_by_id[pid]
    mock_instance.is_validated_project.side_effect = real_client.is_validated_project
    mock_instance.is_oam_project.side_effect = real_client.is_oam_project
    mock_instance.build_project_record.side_effect = real_client.build_project_record
    ctx = mocker.MagicMock(
        __enter__=lambda _self: mock_instance,
        __exit__=mocker.MagicMock(return_value=False),
    )
    mocker.patch("hot_oam_dataset.tm_client.TaskingManagerClient", return_value=ctx)
    return mock_instance


def test_fetch_projects_by_ids_returns_records_for_requested_ids(mocker):
    oam_url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
    details = {
        25317: _detail(oam_url, 25317),
        25320: _detail(oam_url, 25320),
    }
    mock = _make_hand_picked_client(mocker, details)

    projects = fetch_projects_by_ids([25317, 25320])

    assert [p.project_id for p in projects] == [25317, 25320]
    assert mock.fetch_project_full.call_count == 2


def test_fetch_projects_by_ids_warns_but_builds_when_below_validation_threshold(mocker, caplog):
    oam_url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
    detail = _detail(oam_url, 25317)
    detail["percentValidated"] = 40
    _make_hand_picked_client(mocker, {25317: detail})

    with caplog.at_level(logging.WARNING, logger="hot_oam_dataset.tm_client"):
        projects = fetch_projects_by_ids([25317])

    assert len(projects) == 1
    assert projects[0].percent_validated == 40
    assert any("below 95% validation" in rec.message for rec in caplog.records)


def test_fetch_projects_by_ids_warns_but_builds_for_non_oam_imagery(mocker, caplog):
    detail = _detail("https://example.com/tiles/{z}/{x}/{y}", 25317)
    _make_hand_picked_client(mocker, {25317: detail})

    with caplog.at_level(logging.WARNING, logger="hot_oam_dataset.tm_client"):
        projects = fetch_projects_by_ids([25317])

    assert len(projects) == 1
    assert any("does not use OAM imagery" in rec.message for rec in caplog.records)


def _project(project_id: int, name: str = "") -> TaskingManagerProject:
    return TaskingManagerProject(
        project_id=project_id,
        name=name or f"Project {project_id}",
        status="PUBLISHED",
        percent_mapped=100,
        percent_validated=100,
        imagery_url=f"https://tiles.openaerialmap.org/{project_id}/{{z}}/{{x}}/{{y}}",
        area_of_interest={"type": "Polygon", "coordinates": []},
        validated_area={"type": "Polygon", "coordinates": []},
        validated_task_count=1,
        total_task_count=1,
        bbox=[0.0, 0.0, 1.0, 1.0],
        mapping_types=["BUILDINGS"],
        organisation="HOT",
        created="2026-01-01T00:00:00Z",
        country="Nepal",
    )


def _saved_project_ids(output_dir) -> tuple[list[int], list[int]]:
    summary = json.loads((output_dir / "projects_summary.json").read_text())
    features = json.loads((output_dir / "projects_map.geojson").read_text())["features"]
    return (
        [p["project_id"] for p in summary],
        [f["properties"]["project_id"] for f in features],
    )


def test_save_discovery_outputs_keeps_projects_from_earlier_runs(tmp_path):
    save_discovery_outputs([_project(1), _project(2)], tmp_path)
    save_discovery_outputs([_project(3)], tmp_path)

    summary_ids, feature_ids = _saved_project_ids(tmp_path)
    assert summary_ids == [1, 2, 3]
    assert feature_ids == [1, 2, 3]


def test_save_discovery_outputs_replaces_record_for_rebuilt_project(tmp_path):
    save_discovery_outputs([_project(1, name="old name")], tmp_path)
    save_discovery_outputs([_project(1, name="new name")], tmp_path)

    summary = json.loads((tmp_path / "projects_summary.json").read_text())
    assert len(summary) == 1
    assert summary[0]["name"] == "new name"


def test_discover_limit_stops_api_calls(mocker):
    oam_url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"

    mock_instance = mocker.MagicMock()
    mock_instance.iter_projects.return_value = iter(
        [{"projectId": i, "percentValidated": 100} for i in range(1, 100)]
    )
    mock_instance.fetch_project_summary.return_value = _summary(100, ["BUILDINGS"])
    mock_instance.fetch_project_full.side_effect = lambda pid: _detail(oam_url, pid)
    mock_instance.is_validated_project.return_value = True
    mock_instance.is_oam_project.return_value = True
    mock_instance.build_project_record.side_effect = lambda d: mocker.MagicMock(
        project_id=d["projectId"]
    )
    ctx = mocker.MagicMock(
        __enter__=lambda _self: mock_instance,
        __exit__=mocker.MagicMock(return_value=False),
    )
    mocker.patch("hot_oam_dataset.tm_client.TaskingManagerClient", return_value=ctx)

    projects = discover_oam_projects(limit=3)

    assert len(projects) == 3
    assert mock_instance.fetch_project_summary.call_count == 3
    assert mock_instance.fetch_project_full.call_count == 3


def test_tms_prefix_is_stripped_from_imagery_url():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    detail = _detail("tms[22]:https://tiles.openaerialmap.org/abc/{z}/{x}/{y}", 40695)

    project = client.build_project_record(detail)

    assert project.imagery_url == "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"


def test_plain_imagery_url_is_left_alone():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"

    assert client.build_project_record(_detail(url, 1)).imagery_url == url


def test_project_name_is_stripped_of_surrounding_whitespace():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    detail = _detail("https://tiles.openaerialmap.org/abc/{z}/{x}/{y}", 50793)
    detail["projectInfo"]["name"] = "KYC Nairobi Mapping "

    assert client.build_project_record(detail).name == "KYC Nairobi Mapping"


def test_oam_detection_works_on_prefixed_imagery():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert client.is_oam_project({"imagery": "tms[22]:https://tiles.openaerialmap.org/abc"})


def test_listing_does_not_filter_on_imagery(mocker):
    """imagery=custom drops projects whose imagery is stored as tms[zoom]:https://..."""
    client = _make_client(mocker, [_page([1], total_pages=1), _page([], total_pages=1)])

    list(client.iter_projects("ROADS"))

    sent = [call.kwargs["params"] for call in client._client.get.call_args_list]  # noqa: SLF001
    assert sent, "no request was made"
    assert all("imagery" not in params for params in sent)
    assert all(params["mappingTypes"] == "ROADS" for params in sent)


def test_discovery_skips_under_validated_before_fetching(mocker):
    mock = mocker.MagicMock()
    mock.iter_projects.return_value = iter(
        [{"projectId": 1, "percentValidated": 40}, {"projectId": 2, "percentValidated": 100}]
    )
    mock.fetch_project_summary.return_value = _summary(100, ["BUILDINGS"])
    mock.fetch_project_full.side_effect = lambda pid: _detail(
        "https://tiles.openaerialmap.org/a/{z}/{x}/{y}", pid
    )
    mock.is_oam_project.return_value = True
    mock.is_validated_project.return_value = True
    mock.build_project_record.side_effect = TaskingManagerClient.build_project_record.__get__(
        TaskingManagerClient.__new__(TaskingManagerClient)
    )
    ctx = mocker.MagicMock(
        __enter__=lambda _self: mock, __exit__=mocker.MagicMock(return_value=False)
    )
    mocker.patch("hot_oam_dataset.tm_client.TaskingManagerClient", return_value=ctx)

    projects = discover_oam_projects(mapping_type="BUILDINGS")

    assert [p.project_id for p in projects] == [2]
    assert mock.fetch_project_summary.call_count == 1


def test_tms_prefix_with_space_before_zoom_is_stripped():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    detail = _detail("tms [22]:https://tiles.openaerialmap.org/abc/{z}/{x}/{y}", 66864)

    project = client.build_project_record(detail)

    assert project.imagery_url == "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"


def test_hot_imagery_api_openaerialmap_collection_counts_as_oam():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    imagery = (
        "tms[22]:https://api.imagery.hotosm.org/raster/collections/openaerialmap/items/"
        "8494904c/tiles/WebMercatorQuad/{z}/{x}/{y}"
    )
    assert client.is_oam_project({"imagery": imagery})


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("abc123", "abc123"),
        ("Token abc123", "abc123"),
        ("  Token abc123  ", "abc123"),
        ("Tokenabc123", "Tokenabc123"),
    ],
)
def test_token_is_read_with_or_without_scheme(monkeypatch, stored, expected):
    monkeypatch.setenv("TM_TOKEN", stored)
    assert read_tm_token() == expected


def test_missing_token_reads_as_none(monkeypatch):
    monkeypatch.delenv("TM_TOKEN", raising=False)
    assert read_tm_token() is None


def test_client_sends_token_in_authorization_header():
    with TaskingManagerClient(token="abc123") as client:
        assert client._client.headers["Authorization"] == "Token abc123"  # noqa: SLF001


def test_client_without_token_sends_no_authorization_header():
    with TaskingManagerClient(token=None) as client:
        assert "Authorization" not in client._client.headers  # noqa: SLF001


def test_private_project_raises_with_token_hint():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    body = {"Error": "User not permitted: Private Project", "SubCode": "PrivateProject"}
    client._client = httpx.Client(  # noqa: SLF001
        base_url="https://tm.test",
        transport=httpx.MockTransport(lambda _request: httpx.Response(403, json=body)),
    )

    with pytest.raises(PrivateProjectError, match="TM_TOKEN"):
        client.fetch_project_full(40695)


def test_validated_area_covers_only_validated_tasks():
    detail = _detail("", 1, task_statuses=("VALIDATED", "VALIDATED", "MAPPED"))

    area, validated, total = validated_task_area(detail)

    geometry = shape(area)
    assert (validated, total) == (2, 3)
    assert geometry.area == pytest.approx(2.0)
    assert geometry.contains(Point(1.5, 0.5))
    assert not geometry.contains(Point(2.5, 0.5))


def test_validated_area_is_clipped_to_the_project_aoi():
    detail = _detail("", 1, task_statuses=("VALIDATED", "VALIDATED"))
    detail["areaOfInterest"] = mapping(box(0.0, 0.0, 1.5, 1.0))

    area, _, _ = validated_task_area(detail)

    assert shape(area).area == pytest.approx(1.5)
    assert not shape(area).contains(Point(1.8, 0.5))


def test_validated_tasks_touching_the_aoi_only_at_an_edge_count_as_none():
    detail = _detail("", 1, task_statuses=("READY", "VALIDATED"))
    detail["areaOfInterest"] = mapping(box(0.0, 0.0, 1.0, 1.0))

    assert validated_task_area(detail) == ({}, 0, 2)


def test_validated_area_is_empty_without_validated_tasks():
    assert validated_task_area(_detail("", 1, task_statuses=("READY",))) == ({}, 0, 1)


def test_hand_picked_project_without_validated_tasks_is_skipped(mocker, caplog):
    oam_url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"
    details = {
        1: _detail(oam_url, 1, task_statuses=("READY", "MAPPED")),
        2: _detail(oam_url, 2),
    }
    _make_hand_picked_client(mocker, details)

    with caplog.at_level(logging.WARNING, logger="hot_oam_dataset.tm_client"):
        projects = fetch_projects_by_ids([1, 2])

    assert [p.project_id for p in projects] == [2]
    assert any("no validated tasks" in rec.message for rec in caplog.records)


def _make_discovery_client(mocker, detail_for):
    mock = mocker.MagicMock()
    mock.iter_projects.return_value = iter(
        [{"projectId": 1, "percentValidated": 100}, {"projectId": 2, "percentValidated": 100}]
    )
    mock.fetch_project_summary.return_value = _summary(100, ["BUILDINGS"])
    mock.fetch_project_full.side_effect = detail_for
    mock.is_oam_project.return_value = True
    mock.is_validated_project.return_value = True
    mock.build_project_record.side_effect = TaskingManagerClient.build_project_record.__get__(
        TaskingManagerClient.__new__(TaskingManagerClient)
    )
    ctx = mocker.MagicMock(
        __enter__=lambda _self: mock, __exit__=mocker.MagicMock(return_value=False)
    )
    mocker.patch("hot_oam_dataset.tm_client.TaskingManagerClient", return_value=ctx)


def test_discovery_skips_private_projects_and_keeps_going(mocker):
    oam_url = "https://tiles.openaerialmap.org/a/{z}/{x}/{y}"

    def detail_for(pid):
        if pid == 1:
            raise PrivateProjectError("TM refused project 1 (403)")
        return _detail(oam_url, pid)

    _make_discovery_client(mocker, detail_for)

    assert [p.project_id for p in discover_oam_projects()] == [2]


def test_discovery_skips_projects_without_validated_tasks(mocker):
    oam_url = "https://tiles.openaerialmap.org/a/{z}/{x}/{y}"
    statuses = {1: ("BADIMAGERY",), 2: ("VALIDATED",)}
    _make_discovery_client(mocker, lambda pid: _detail(oam_url, pid, statuses[pid]))

    assert [p.project_id for p in discover_oam_projects()] == [2]
