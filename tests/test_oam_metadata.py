import httpx
import pytest

from hot_oam_dataset import oam_metadata
from hot_oam_dataset.oam_metadata import oam_imagery

TMS = "https://tiles.openaerialmap.org/a/0/b/{z}/{x}/{y}"
OTHER = "https://tiles.openaerialmap.org/other/{z}/{x}/{y}"
HOT_ITEM = "8494904c-973b-49d6-bda8-f35f5538bbe1"


def _record(title, tms):
    return {
        "title": title,
        "acquisition_start": "2025-10-22T00:00:00.000Z",
        "properties": {"tms": tms},
    }


def _response(url, payload):
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _search_returns(mocker, records):
    payload = {"meta": {"found": len(records)}, "results": records}
    return mocker.patch.object(
        oam_metadata.httpx, "get", return_value=_response(oam_metadata.OAM_META_URL, payload)
    )


def test_the_record_with_the_project_tile_url_is_returned(mocker):
    get = _search_returns(mocker, [_record("Other", OTHER), _record("Freetown", TMS)])

    imagery = oam_imagery(TMS, [-13.3, 8.4, -13.1, 8.5])

    assert imagery.title == "Freetown"
    assert imagery.acquired.isoformat() == "2025-10-22T00:00:00+00:00"
    assert get.call_args.kwargs["params"]["bbox"] == "-13.3,8.4,-13.1,8.5"


def test_a_tile_url_with_an_image_suffix_matches_its_record(mocker):
    _search_returns(mocker, [_record("Kathmandu", TMS)])

    assert oam_imagery(f"{TMS}.png", [0, 0, 1, 1]).title == "Kathmandu"


def test_every_results_page_is_read(mocker):
    first = {"meta": {"found": 2}, "results": [_record("Other", OTHER)]}
    second = {"meta": {"found": 2}, "results": [_record("Freetown", TMS)]}
    mocker.patch.object(
        oam_metadata.httpx,
        "get",
        side_effect=[
            _response(oam_metadata.OAM_META_URL, first),
            _response(oam_metadata.OAM_META_URL, second),
        ],
    )

    assert oam_imagery(TMS, [0, 0, 1, 1]).title == "Freetown"


@pytest.mark.parametrize(("records", "found"), [([], 0), ([TMS, TMS], 2)])
def test_missing_or_ambiguous_imagery_is_an_error(mocker, records, found):
    _search_returns(mocker, [_record("Freetown", tms) for tms in records])

    with pytest.raises(LookupError, match=f"found {found}"):
        oam_imagery(TMS, [0, 0, 1, 1])


def test_hot_imagery_api_tiles_read_their_stac_item(mocker):
    item = {
        "properties": {
            "title": "Rimuka",
            "datetime": None,
            "start_datetime": "2026-07-11T00:00:00Z",
        }
    }
    get = mocker.patch.object(
        oam_metadata.httpx, "get", return_value=_response("https://api.imagery.hotosm.org", item)
    )
    tiles = (
        "https://api.imagery.hotosm.org/raster/collections/openaerialmap/items/"
        f"{HOT_ITEM}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}?assets=visual"
    )

    imagery = oam_imagery(tiles, [0, 0, 1, 1])

    assert imagery.title == "Rimuka"
    assert imagery.acquired.isoformat() == "2026-07-11T00:00:00+00:00"
    assert get.call_args.args[0].endswith(f"/stac/collections/openaerialmap/items/{HOT_ITEM}")
