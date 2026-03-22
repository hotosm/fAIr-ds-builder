from hot_oam_dataset.tm_client import (
    TaskingManagerClient,
    discover_oam_building_projects,
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
    assert list(client.iter_custom_imagery_project_ids()) == [1, 2, 3, 4, 5]


def test_iter_ids_generator_is_lazy(mocker):
    client = _make_client(mocker, [_page(list(range(1, 11)), total_pages=5)])

    gen = client.iter_custom_imagery_project_ids()
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


def test_is_validated_building_project_passes():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert client.is_validated_building_project(_summary(95, ["BUILDINGS"]))


def test_is_validated_building_project_below_threshold():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_validated_building_project(_summary(94, ["BUILDINGS"]))


def test_is_validated_building_project_wrong_type():
    client = TaskingManagerClient.__new__(TaskingManagerClient)
    assert not client.is_validated_building_project(_summary(100, ["ROADS"]))


def test_discover_limit_stops_api_calls(mocker):
    oam_url = "https://tiles.openaerialmap.org/abc/{z}/{x}/{y}"

    mock_instance = mocker.MagicMock()
    mock_instance.iter_custom_imagery_project_ids.return_value = iter(range(1, 100))
    mock_instance.fetch_project_summary.return_value = _summary(100, ["BUILDINGS"])
    mock_instance.fetch_project_full.side_effect = lambda pid: _detail(oam_url, pid)
    mock_instance.is_validated_building_project.return_value = True
    mock_instance.is_oam_project.return_value = True
    mock_instance.build_project_record.side_effect = lambda d: mocker.MagicMock(
        project_id=d["projectId"]
    )
    ctx = mocker.MagicMock(
        __enter__=lambda s: mock_instance,
        __exit__=mocker.MagicMock(return_value=False),
    )
    mocker.patch("hot_oam_dataset.tm_client.TaskingManagerClient", return_value=ctx)

    projects = discover_oam_building_projects(limit=3)

    assert len(projects) == 3
    assert mock_instance.fetch_project_summary.call_count == 3
    assert mock_instance.fetch_project_full.call_count == 3
