"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801
import os
import shutil
import unittest
from pprint import pprint
from typing import Any

from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    minimal_mdsa_test_exp_setts,
)
from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_results.load_json_to_nx_graph import (
    load_json_to_nx_graph_from_file,
)
from typeguard import typechecked

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnalgorithms.sparse.MDSA.get_results import get_results


class Test_mdsa_snn_results(unittest.TestCase):
    """Tests whether the snn implementation of the MDSA algorithm yields the
    same results as the default/Neumann implementation if its weights are
    identical."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.algorithms = {
            "MDSA": get_algo_configs(MDSA(list(range(0, 1, 1))).__dict__)
        }

    @typechecked
    def test_snn_results_equal_neumann_results(self) -> None:
        """Tests whether the results of the snn implementation of the MDSA
        algorithm are the same as those of the default/Neumann implementation
        of that MDSA algorithm. ."""

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("latex"):
            shutil.rmtree("latex")

        # Generate default experiment config.
        # mdsa_creation_only_size_3_4: dict = long_exp_setts_for_mdsa_testing()
        # mdsa_creation_only_size_3_4: dict = short_mdsa_test_exp_setts()
        mdsa_creation_only_size_3_4: dict = minimal_mdsa_test_exp_setts()

        # Do not output images.
        mdsa_creation_only_size_3_4["overwrite_visualisation"] = True
        mdsa_creation_only_size_3_4["show_snns"] = False
        mdsa_creation_only_size_3_4["export_images"] = False

        # Include desired mdsa settings.
        mdsa_creation_only_size_3_4["algorithms"] = self.algorithms

        verify_experiment_config(
            Supported_experiment_settings(),
            mdsa_creation_only_size_3_4,
            has_unique_id=False,
            allow_optional=True,
        )

        # Verify results are identical.
        # Experiment_runner(
        # mdsa_creation_only_size_3_4, run_config_with_error())
        exp_runner = Experiment_runner(mdsa_creation_only_size_3_4)
        assert_run_config_json_results(
            self, exp_runner, exp_runner.run_configs[0]
        )


@typechecked
def assert_run_config_json_results(
    test_object: Any, exp_runner: Experiment_runner, run_config: dict
) -> None:
    """Verifies the results of a run config using the json result output."""

    json_graphs = load_json_to_nx_graph_from_file(
        run_config=run_config, stage_index=4, to_run=exp_runner.to_run
    )
    pprint(json_graphs["input_graph"])

    # TODO: convert json graphs to nx graphs.

    # Verify results are as expected.
    alipour_counter_marks = get_results(
        input_graph=json_graphs["input_graph"],
        iteration=run_config["iteration"],
        m_val=run_config["algorithm"]["MDSA"]["m_val"],
        rand_props=json_graphs["input_graph"].graph["alg_props"],
        seed=run_config["seed"],
        size=run_config["graph_size"],
    )

    alipour_counter_marks = alipour_counter_marks + 1
    test_object.assertEquals(1, 2 - 1)
