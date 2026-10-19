import pytest

from hot_oam_dataset import main as cli


def test_tag_values_reach_the_build_config(tmp_path, mocker):
    mocker.patch.object(cli, "fetch_projects_by_ids", return_value=[mocker.MagicMock()])
    mocker.patch.object(cli, "_print_projects_table")
    build = mocker.patch.object(cli, "build_all_projects", return_value=[])

    cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--project-ids",
            "50661",
            "--feature-type",
            "natural",
            "--tag-values",
            "tree",
            "tree_group",
        ]
    )

    config = build.call_args.args[1]
    assert config.tag_values == ["tree", "tree_group"]
    assert config.feature_spec.raw_data_filters() == {
        "tags": {"all_geometry": {"join_or": {"natural": ["tree", "tree_group"]}}}
    }


def test_required_tags_narrow_the_request_to_features_carrying_both(tmp_path, mocker):
    mocker.patch.object(cli, "fetch_projects_by_ids", return_value=[mocker.MagicMock()])
    mocker.patch.object(cli, "_print_projects_table")
    build = mocker.patch.object(cli, "build_all_projects", return_value=[])

    cli.main(
        [
            "--output-dir",
            str(tmp_path),
            "--project-ids",
            "50793",
            "--feature-type",
            "surface",
            "--require-tags",
            "highway",
        ]
    )

    assert build.call_args.args[1].feature_spec.raw_data_filters() == {
        "tags": {"all_geometry": {"join_and": {"surface": [], "highway": []}}}
    }


@pytest.mark.parametrize("arguments", [["--negative-ratio", "-1"], ["--zoom", "nineteen"]])
def test_out_of_range_numbers_are_rejected(arguments):
    with pytest.raises(SystemExit):
        cli.parse_args(arguments)


def _config(tmp_path, text):
    path = tmp_path / "dataset.yaml"
    path.write_text(text)
    return str(path)


def test_config_file_supplies_options_and_flags_override_it(tmp_path):
    config = _config(
        tmp_path,
        "project_ids: 25317,25320\nfeature_type: natural\ntag_values: [tree, tree_group]\n"
        "zoom: 20\nburn_binary_masks: false\n",
    )

    args = cli.parse_args(["--config", config, "--zoom", "19"])

    assert args.project_ids == [25317, 25320]
    assert args.tag_values == ["tree", "tree_group"]
    assert args.burn_binary_masks is False
    assert args.zoom == 19


@pytest.mark.parametrize(
    "text", ["geometry_types: [area]\n", "not_an_option: 1\n", "split_block_size: 8\n"]
)
def test_config_file_values_are_validated_like_flags(tmp_path, text):
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", _config(tmp_path, text)])


def test_missing_config_file_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", str(tmp_path / "absent.yaml")])


def test_config_card_section_becomes_card_text_and_not_a_flag(tmp_path):
    config = _config(
        tmp_path,
        "zoom: 20\ncard:\n  repo_id: hotosm/trees\n  title: Trees\n  label_noun: trees\n"
        "  tasks: [detection]\n  summary: Tree data.\n  selection: Points.\n"
        "  limitations: [Estimates.]\n",
    )

    args = cli.parse_args(["--config", config])

    assert args.zoom == 20
    assert args.card_text.repo_id == "hotosm/trees"
    assert args.card_text.limitations == ("Estimates.",)


def test_config_without_card_section_leaves_card_text_unset(tmp_path):
    args = cli.parse_args(["--config", _config(tmp_path, "zoom: 20\n")])

    assert args.card_text is None


def test_config_split_section_sets_the_split_settings(tmp_path):
    config = _config(tmp_path, "split:\n  buffer_meters: 900\n  regions_per_project: 12\n")

    settings = cli.parse_args(["--config", config]).split_settings

    assert settings.buffer_meters == 900
    assert settings.regions_per_project == 12
    assert settings.exclude_shared_features is False


def test_config_split_section_with_an_unknown_setting_is_rejected(tmp_path, capsys):
    config = _config(tmp_path, "split:\n  block_size: 8\n")

    with pytest.raises(SystemExit):
        cli.parse_args(["--config", config])
    assert "unknown split settings" in capsys.readouterr().err


@pytest.mark.parametrize(("extra", "steps_run"), [(["--all"], True), ([], False)])
def test_all_runs_splits_parquet_and_stac_after_the_build(tmp_path, mocker, extra, steps_run):
    mocker.patch.object(cli, "fetch_projects_by_ids", return_value=[mocker.MagicMock()])
    mocker.patch.object(cli, "_print_projects_table")
    mocker.patch.object(cli, "build_all_projects", return_value=[])
    calls = mocker.MagicMock()
    mocker.patch.object(cli, "generate_splits", calls.splits)
    mocker.patch.object(cli, "generate_parquet", calls.parquet)
    mocker.patch.object(cli, "generate_stac", calls.stac)

    cli.main(["--output-dir", str(tmp_path), "--project-ids", "50661", *extra])

    expected = ["splits", "parquet", "stac"] if steps_run else []
    assert [name for name, _, _ in calls.mock_calls] == expected


def test_config_fair_section_sets_who_registers_the_dataset(tmp_path):
    config = _config(tmp_path, "fair:\n  user_id: hot\n  provider: hot\n")

    settings = cli.parse_args(["--config", config]).fair_settings

    assert (settings.user_id, settings.provider) == ("hot", "hot")


def test_publish_stac_without_card_and_fair_sections_is_a_usage_error(tmp_path, mocker):
    publish = mocker.patch.object(cli, "publish_to_fair")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--output-dir", str(tmp_path), "--publish-stac", "https://stac.example.org"])

    assert exit_info.value.code == 2
    publish.assert_not_called()


def test_update_root_catalog_without_a_card_section_is_a_usage_error(tmp_path, mocker):
    update = mocker.patch.object(cli, "update_root_catalog")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--output-dir", str(tmp_path), "--update-root-catalog", "hotosm/fAIr-guru-stac"])

    assert exit_info.value.code == 2
    update.assert_not_called()


def test_all_combined_with_a_single_step_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(["--output-dir", str(tmp_path), "--all", "--generate-stac"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--all", "--compress-raw"],
        ["--publish-stac", "https://stac.example.org", "--push-to-hub"],
        ["--update-root-catalog", "hotosm/fAIr-guru-stac", "--push-to-hub"],
    ],
)
def test_steps_that_must_run_alone_are_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit):
        cli.parse_args(["--output-dir", str(tmp_path), *arguments])
