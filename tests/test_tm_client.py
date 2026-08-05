import json
import logging

from hot_oam_dataset.tm_client import (
    TaskingManagerClient,
    TaskingManagerProject,
    discover_oam_projects,
    fetch_projects_by_ids,
    save_discovery_outputs,
)


def _page(project_ids: list[int], total_pages: int) -> dict:
    return {
        "results": [{"projectId": pid} for pid in project_ids],
        "pagination": {"pages": total_pages},
    }


def _summary(percent_validated: int, mapping_types: list[str]) -> dict:
    return {"percentValidated": percent_validated, "mappingTypes": mapping_types}


def _detail(imagery: str, project_id: int) -> dict:
    return {
        "projectId": project_id,
        "imagery": imagery,
        "projectInfo": {"name": f"Project {project_id}"},
        "status": "PUBLISHED",
        "percentMapped": 100,
        "percentValidated": 100,
        "mappingTypes": ["BUILDINGS"],
        "areaOfInterest": {"type": "Polygon", "coordinates": []},
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
    assert list(client.iter_custom_imagery_project_ids("BUILDINGS")) == [1, 2, 3, 4, 5]


def test_iter_ids_generator_is_lazy(mocker):
    client = _make_client(mocker, [_page(list(range(1, 11)), total_pages=5)])

    gen = client.iter_custom_imagery_project_ids("BUILDINGS")
    first_three = [next(gen) for _ in range(3)]

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
    mock_instance.iter_custom_imagery_project_ids.return_value = iter(range(1, 100))
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
