"""A static STAC catalogue in its own Hub repo, with each dataset release linked as a child."""

import json
import logging
import re

import pystac
from huggingface_hub import CommitOperationAdd, HfApi

from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.hf_publisher import HUB_DATASETS_URL, release_stac_url
from hot_oam_dataset.stac_export import STAC_DIRNAME

logger = logging.getLogger(__name__)

ROOT_CATALOG_FILENAME = "catalog.json"
ROOT_CATALOG_TITLE = "HOT training datasets"
ROOT_CATALOG_DESCRIPTION = (
    "Training datasets built from HOT Tasking Manager projects and their OpenAerialMap imagery. "
    "Each child is the STAC Collection of one dataset release on Hugging Face."
)
STAC_BROWSER_URL = "https://radiantearth.github.io/stac-browser"
RELEASE_URL_PATTERN = re.compile(
    rf"^(?P<dataset>{re.escape(HUB_DATASETS_URL)}/[^/]+/[^/]+)/raw/(?P<tag>[^/]+)/"
)


def root_catalog_url(root_repo_id: str) -> str:
    return f"{HUB_DATASETS_URL}/{root_repo_id}/raw/main/{ROOT_CATALOG_FILENAME}"


def release_collection_url(repo_id: str, dataset_version: str) -> str:
    return f"{release_stac_url(repo_id, dataset_version, 'raw')}/collection.json"


def new_root_catalog(root_repo_id: str) -> pystac.Catalog:
    catalog = pystac.Catalog(
        id=root_repo_id.split("/")[-1],
        title=ROOT_CATALOG_TITLE,
        description=ROOT_CATALOG_DESCRIPTION,
    )
    catalog.set_self_href(root_catalog_url(root_repo_id))
    return catalog


def link_release(
    catalog: pystac.Catalog, repo_id: str, dataset_version: str, title: str
) -> pystac.Catalog:
    """Link the release as a child, replacing any other release of the same dataset."""
    dataset_prefix = f"{HUB_DATASETS_URL}/{repo_id}/"
    catalog.links = [
        link
        for link in catalog.links
        if not (link.rel == pystac.RelType.CHILD and link.href.startswith(dataset_prefix))
    ]
    catalog.add_link(
        pystac.Link(
            pystac.RelType.CHILD,
            release_collection_url(repo_id, dataset_version),
            media_type=pystac.MediaType.JSON,
            title=title,
        )
    )
    children = sorted(catalog.get_links(pystac.RelType.CHILD), key=lambda link: link.title or "")
    catalog.links = [link for link in catalog.links if link.rel != pystac.RelType.CHILD] + children
    return catalog


def _release_row(link: pystac.Link) -> str:
    release = RELEASE_URL_PATTERN.match(link.href)
    if not release:
        raise ValueError(f"Child link {link.href} is not a Hugging Face dataset release")
    return f"| [{link.title}]({release['dataset']}) | {release['tag']} |"


def root_catalog_card(catalog: pystac.Catalog, root_repo_id: str) -> str:
    url = root_catalog_url(root_repo_id)
    browser_url = f"{STAC_BROWSER_URL}/#/external/{url.removeprefix('https://')}"
    lines = [
        "---",
        "license:",
        "- cc-by-4.0",
        "- odbl",
        "---",
        "",
        f"# {ROOT_CATALOG_TITLE}",
        "",
        ROOT_CATALOG_DESCRIPTION,
        "",
        f"Catalogue URL, for STAC Browser or the QGIS STAC panel: <{url}>",
        "",
        f"[Open in STAC Browser]({browser_url})",
        "",
        "| Dataset | Release |",
        "| --- | --- |",
        *(_release_row(link) for link in catalog.get_links(pystac.RelType.CHILD)),
        "",
    ]
    return "\n".join(lines)


def _hub_root_catalog(api: HfApi, root_repo_id: str) -> pystac.Catalog:
    if not api.file_exists(root_repo_id, ROOT_CATALOG_FILENAME, repo_type="dataset"):
        logger.info("No %s in %s yet, starting one", ROOT_CATALOG_FILENAME, root_repo_id)
        return new_root_catalog(root_repo_id)
    # Forced, so a failed request raises and never falls back to a stale cached copy
    path = api.hf_hub_download(
        root_repo_id, ROOT_CATALOG_FILENAME, repo_type="dataset", force_download=True
    )
    catalog = pystac.Catalog.from_file(path)
    # Its root link names the catalogue itself, which pystac would otherwise download again
    catalog.set_root(catalog)
    catalog.set_self_href(root_catalog_url(root_repo_id))
    return catalog


def update_root_catalog(root_repo_id: str, card_text: CardText, dataset_version: str) -> None:
    """Link the dataset's Hub release in the root catalogue and upload it with its card."""
    api = HfApi()
    collection_path = f"{STAC_DIRNAME}/collection.json"
    if not api.file_exists(
        card_text.repo_id, collection_path, repo_type="dataset", revision=f"v{dataset_version}"
    ):
        msg = (
            f"{card_text.repo_id} release v{dataset_version} has no {collection_path}; "
            "run --push-to-hub with this version first"
        )
        raise ValueError(msg)

    catalog = link_release(
        _hub_root_catalog(api, root_repo_id), card_text.repo_id, dataset_version, card_text.title
    )
    catalog.validate()
    api.create_repo(root_repo_id, repo_type="dataset", exist_ok=True)
    catalogue_json = json.dumps(catalog.to_dict(transform_hrefs=False), indent=2) + "\n"
    api.create_commit(
        repo_id=root_repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(ROOT_CATALOG_FILENAME, catalogue_json.encode()),
            CommitOperationAdd("README.md", root_catalog_card(catalog, root_repo_id).encode()),
        ],
        commit_message=f"Link {card_text.repo_id} v{dataset_version}",
    )
    logger.info("Root catalogue updated at %s", root_catalog_url(root_repo_id))
