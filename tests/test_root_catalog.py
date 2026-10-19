import json
import re

import pystac
import pytest

from hot_oam_dataset import root_catalog
from hot_oam_dataset.dataset_card import CardText
from hot_oam_dataset.root_catalog import (
    link_release,
    new_root_catalog,
    root_catalog_card,
    update_root_catalog,
)

ROOT_REPO = "hotosm/fAIr-guru-stac"
TREES = "hotosm/fAIr-guru-td-trees"
SOLAR = "hotosm/fAIr-guru-td-solar"
CARD = CardText.from_mapping(
    {
        "repo_id": TREES,
        "title": "fAIr Open Mapping Gurus: trees",
        "label_noun": "trees",
        "tasks": ["detection"],
        "summary": "Tree training data from Freetown.",
        "selection": "Trees mapped as points and tree groups as polygons.",
        "limitations": [],
    }
)


def _child_hrefs(catalog: pystac.Catalog) -> list[str]:
    return [link.href for link in catalog.get_links(pystac.RelType.CHILD)]


def _release_url(repo_id: str, version: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/raw/v{version}/stac/collection.json"


def test_a_release_is_linked_as_a_child_at_its_tag():
    catalog = link_release(new_root_catalog(ROOT_REPO), TREES, "0.1.0", CARD.title)

    assert _child_hrefs(catalog) == [_release_url(TREES, "0.1.0")]
    assert catalog.get_self_href() == (
        "https://huggingface.co/datasets/hotosm/fAIr-guru-stac/raw/main/catalog.json"
    )
    catalog.validate()


def test_a_new_release_replaces_the_old_one_and_keeps_other_datasets_in_title_order():
    catalog = new_root_catalog(ROOT_REPO)
    link_release(catalog, TREES, "0.1.0", "fAIr Open Mapping Gurus: trees")
    link_release(catalog, SOLAR, "0.1.0", "fAIr Open Mapping Gurus: solar")

    link_release(catalog, TREES, "0.2.0", "fAIr Open Mapping Gurus: trees")

    assert _child_hrefs(catalog) == [_release_url(SOLAR, "0.1.0"), _release_url(TREES, "0.2.0")]


def test_a_dataset_whose_name_extends_another_keeps_its_own_link():
    catalog = link_release(new_root_catalog(ROOT_REPO), TREES, "0.1.0", "trees")

    link_release(catalog, f"{TREES}-v2", "0.1.0", "trees v2")

    assert _child_hrefs(catalog) == [
        _release_url(TREES, "0.1.0"),
        _release_url(f"{TREES}-v2", "0.1.0"),
    ]


def test_the_card_lists_each_linked_release():
    catalog = link_release(new_root_catalog(ROOT_REPO), TREES, "0.1.0", CARD.title)

    card = root_catalog_card(catalog, ROOT_REPO)

    assert f"| [{CARD.title}](https://huggingface.co/datasets/{TREES}) | v0.1.0 |" in card


@pytest.fixture
def hub(mocker):
    api = mocker.MagicMock()
    mocker.patch.object(root_catalog, "HfApi", return_value=api)
    return api


def _hub_has(hub, root_catalog_exists: bool) -> None:
    hub.file_exists.side_effect = lambda repo_id, *_, **__: (
        repo_id != ROOT_REPO or root_catalog_exists
    )


def _committed_catalog(hub) -> dict:
    operations = hub.create_commit.call_args.kwargs["operations"]
    catalogue = next(op for op in operations if op.path_in_repo == "catalog.json")
    return json.loads(catalogue.path_or_fileobj)


def test_a_release_without_a_stac_collection_is_rejected_before_any_upload(hub):
    hub.file_exists.return_value = False

    with pytest.raises(ValueError, match=re.escape("release v0.1.0 has no stac/collection.json")):
        update_root_catalog(ROOT_REPO, CARD, "0.1.0")

    hub.create_commit.assert_not_called()


def test_the_first_release_creates_the_root_catalogue(hub):
    _hub_has(hub, root_catalog_exists=False)

    update_root_catalog(ROOT_REPO, CARD, "0.1.0")

    hub.hf_hub_download.assert_not_called()
    hub.create_repo.assert_called_once_with(ROOT_REPO, repo_type="dataset", exist_ok=True)
    children = [link["href"] for link in _committed_catalog(hub)["links"] if link["rel"] == "child"]
    assert children == [_release_url(TREES, "0.1.0")]


def test_a_failed_download_of_the_root_catalogue_uploads_nothing(hub):
    _hub_has(hub, root_catalog_exists=True)
    hub.hf_hub_download.side_effect = ValueError("Force download failed")

    with pytest.raises(ValueError, match="Force download failed"):
        update_root_catalog(ROOT_REPO, CARD, "0.1.0")

    hub.create_commit.assert_not_called()


def test_releases_already_in_the_hub_catalogue_are_kept(hub, tmp_path):
    _hub_has(hub, root_catalog_exists=True)
    existing = link_release(new_root_catalog(ROOT_REPO), SOLAR, "0.1.0", "solar")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(existing.to_dict(transform_hrefs=False)))
    hub.hf_hub_download.return_value = str(path)

    update_root_catalog(ROOT_REPO, CARD, "0.1.0")

    children = [link["href"] for link in _committed_catalog(hub)["links"] if link["rel"] == "child"]
    assert children == [_release_url(TREES, "0.1.0"), _release_url(SOLAR, "0.1.0")]
