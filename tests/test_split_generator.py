import json

import pytest

from hot_oam_dataset.split_generator import (
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    ProjectCluster,
    assign_splits,
    build_clusters,
    compute_split_stats,
    generate_splits,
    load_splits,
)


def _write_stats(path, projects):
    path.write_text(json.dumps({"projects": projects}))


def _write_summary(path, projects):
    path.write_text(json.dumps(projects))


@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path


def test_assign_splits_single_large_cluster():
    clusters = [
        ProjectCluster(imagery_url="url_a", project_ids=[1, 2, 3], total_tiles=8000),
        ProjectCluster(imagery_url="url_b", project_ids=[4], total_tiles=1000),
        ProjectCluster(imagery_url="url_c", project_ids=[5], total_tiles=1000),
    ]
    assignments = assign_splits(clusters)

    assert assignments[1] == SPLIT_TRAIN
    assert assignments[2] == SPLIT_TRAIN
    assert assignments[3] == SPLIT_TRAIN
    assert assignments[4] != SPLIT_TRAIN
    assert assignments[5] != SPLIT_TRAIN
    assert assignments[4] != assignments[5]


def test_assign_splits_all_projects_assigned():
    clusters = [
        ProjectCluster(imagery_url=f"url_{i}", project_ids=[i], total_tiles=100) for i in range(10)
    ]
    assignments = assign_splits(clusters)

    assert len(assignments) == 10
    assert all(s in SPLIT_NAMES for s in assignments.values())


def test_assign_splits_empty():
    assert assign_splits([]) == {}


def test_shared_imagery_url_stays_together(dataset_dir):
    _write_stats(
        dataset_dir / "dataset_stats.json",
        [
            {"project_id": 1, "valid_pairs": 500},
            {"project_id": 2, "valid_pairs": 500},
            {"project_id": 3, "valid_pairs": 200},
        ],
    )
    _write_summary(
        dataset_dir / "projects_summary.json",
        [
            {"project_id": 1, "imagery_url": "https://oam.org/shared"},
            {"project_id": 2, "imagery_url": "https://oam.org/shared"},
            {"project_id": 3, "imagery_url": "https://oam.org/unique"},
        ],
    )

    clusters = build_clusters(dataset_dir)

    shared_cluster = next(c for c in clusters if len(c.project_ids) == 2)
    assert set(shared_cluster.project_ids) == {1, 2}
    assert shared_cluster.total_tiles == 1000


def test_generate_and_load_roundtrip(dataset_dir):
    _write_stats(
        dataset_dir / "dataset_stats.json",
        [{"project_id": i, "valid_pairs": 100} for i in range(1, 11)],
    )
    _write_summary(
        dataset_dir / "projects_summary.json",
        [{"project_id": i, "imagery_url": f"https://oam.org/{i}"} for i in range(1, 11)],
    )

    assignments = generate_splits(dataset_dir)
    loaded = load_splits(dataset_dir)

    assert assignments == loaded
    assert len(loaded) == 10


def test_load_splits_missing_file(dataset_dir):
    assert load_splits(dataset_dir) == {}


def test_split_ratios_approximate():
    clusters = [
        ProjectCluster(imagery_url=f"url_{i}", project_ids=[i], total_tiles=100) for i in range(100)
    ]
    assignments = assign_splits(clusters)

    total = 10000
    train_tiles = sum(100 for pid, s in assignments.items() if s == SPLIT_TRAIN)
    val_tiles = sum(100 for pid, s in assignments.items() if s == SPLIT_VALIDATION)
    test_tiles = sum(100 for pid, s in assignments.items() if s == SPLIT_TEST)

    assert 0.70 * total <= train_tiles <= 0.90 * total
    assert 0.05 * total <= val_tiles <= 0.20 * total
    assert 0.05 * total <= test_tiles <= 0.20 * total


def test_dominant_cluster_forced_to_train():
    clusters = [
        ProjectCluster(imagery_url="dominant", project_ids=[1], total_tiles=6000),
        ProjectCluster(imagery_url="url_a", project_ids=[2], total_tiles=500),
        ProjectCluster(imagery_url="url_b", project_ids=[3], total_tiles=500),
        ProjectCluster(imagery_url="url_c", project_ids=[4], total_tiles=500),
        ProjectCluster(imagery_url="url_d", project_ids=[5], total_tiles=500),
    ]
    assignments = assign_splits(clusters)

    assert assignments[1] == SPLIT_TRAIN


def test_build_clusters_excludes_projects_without_stats(dataset_dir):
    _write_stats(
        dataset_dir / "dataset_stats.json",
        [{"project_id": 1, "valid_pairs": 100}],
    )
    _write_summary(
        dataset_dir / "projects_summary.json",
        [
            {"project_id": 1, "imagery_url": "https://oam.org/1"},
            {"project_id": 99, "imagery_url": "https://oam.org/99"},
        ],
    )

    clusters = build_clusters(dataset_dir)
    all_pids = [pid for c in clusters for pid in c.project_ids]

    assert 1 in all_pids
    assert 99 not in all_pids


def test_compute_split_stats_includes_features_and_countries(dataset_dir):
    _write_stats(
        dataset_dir / "dataset_stats.json",
        [
            {
                "project_id": 1,
                "valid_pairs": 800,
                "total_label_features": 5000,
                "country": "Kenya",
            },
            {
                "project_id": 2,
                "valid_pairs": 100,
                "total_label_features": 600,
                "country": "Peru",
            },
            {
                "project_id": 3,
                "valid_pairs": 100,
                "total_label_features": 400,
                "country": "Ghana",
            },
        ],
    )
    _write_summary(
        dataset_dir / "projects_summary.json",
        [
            {"project_id": 1, "imagery_url": "https://oam.org/1"},
            {"project_id": 2, "imagery_url": "https://oam.org/2"},
            {"project_id": 3, "imagery_url": "https://oam.org/3"},
        ],
    )

    generate_splits(dataset_dir)
    stats = compute_split_stats(dataset_dir)

    assert stats is not None
    total_features = sum(s["features"] for s in stats.values())
    assert total_features == 6000

    for split_name in SPLIT_NAMES:
        info = stats[split_name]
        assert "features" in info
        assert "countries" in info
        assert isinstance(info["countries"], list)


def test_compute_split_stats_returns_none_without_splits(dataset_dir):
    assert compute_split_stats(dataset_dir) is None
