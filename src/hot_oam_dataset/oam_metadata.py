"""Source imagery facts from the OpenAerialMap catalogue."""

import re
from dataclasses import dataclass
from datetime import datetime

import httpx

OAM_META_URL = "https://api.openaerialmap.org/meta"
OAM_SEARCH_LIMIT = 100
# Imagery served by the HOT imagery API carries its STAC item id in the tile URL
HOT_IMAGERY_TILES = re.compile(
    r"^https://api\.imagery\.hotosm\.org/raster/collections/(?P<collection>[^/]+)"
    r"/items/(?P<item>[^/]+)/"
)
HOT_IMAGERY_ITEM_URL = "https://api.imagery.hotosm.org/stac/collections/{collection}/items/{item}"
TILE_SUFFIX = re.compile(r"\.(png|jpe?g)$")


@dataclass(frozen=True)
class OamImagery:
    title: str
    acquired: datetime


def _hot_imagery_item(collection: str, item: str) -> OamImagery:
    url = HOT_IMAGERY_ITEM_URL.format(collection=collection, item=item)
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    properties = response.json()["properties"]
    return OamImagery(
        title=properties["title"],
        acquired=datetime.fromisoformat(properties["datetime"] or properties["start_datetime"]),
    )


def _catalogue_records(bbox: list[float]) -> list[dict]:
    """Every catalogue record intersecting bbox, read page by page."""
    params = {"bbox": ",".join(map(str, bbox)), "limit": OAM_SEARCH_LIMIT}
    records: list[dict] = []
    page = 1
    while True:
        response = httpx.get(OAM_META_URL, params=params | {"page": page}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        records += payload["results"]
        if len(records) >= payload["meta"]["found"] or not payload["results"]:
            return records
        page += 1


def oam_imagery(tms_url: str, bbox: list[float]) -> OamImagery:
    """The OpenAerialMap record of the imagery a project's tile URL serves."""
    hot_tiles = HOT_IMAGERY_TILES.match(tms_url)
    if hot_tiles:
        return _hot_imagery_item(hot_tiles["collection"], hot_tiles["item"])

    wanted = TILE_SUFFIX.sub("", tms_url)
    matches = [
        record
        for record in _catalogue_records(bbox)
        if TILE_SUFFIX.sub("", record.get("properties", {}).get("tms", "")) == wanted
    ]
    if len(matches) != 1:
        raise LookupError(f"Expected one OpenAerialMap record for {tms_url}, found {len(matches)}")
    record = matches[0]
    return OamImagery(
        title=record["title"],
        acquired=datetime.fromisoformat(record["acquisition_start"]),
    )
