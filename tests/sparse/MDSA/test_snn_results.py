"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801
import copy
import os
import shutil
import unittest
from typing import Any, Dict

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
        mdsa_creation_only_size_3_4["overwrite_snn_propagation"] = True
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

        for run_config in exp_runner.run_configs:
            assert_run_config_json_results(self, exp_runner, run_config)


@typechecked
def assert_run_config_json_results(
    test_object: Any, exp_runner: Experiment_runner, run_config: dict
) -> None:
    """Verifies the results of a run config using the json result output."""

    nx_graphs = load_json_to_nx_graph_from_file(
        run_config=run_config, stage_index=4, to_run=exp_runner.to_run
    )

    # Verify results are as expected.
    expected_nodenames: Dict[str, int] = get_results(
        input_graph=nx_graphs["input_graph"],
        iteration=run_config["iteration"],
        m_val=run_config["algorithm"]["MDSA"]["m_val"],
        rand_props=nx_graphs["input_graph"].graph["alg_props"],
        seed=run_config["seed"],
        size=run_config["graph_size"],
    )

    for graph_name, snn_graph in nx_graphs.items():

        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:

            # Verify the SNN graphs have completed simulation stage 2.
            test_object.assertTrue(
                # pylint: disable=R1733
                2
                in nx_graphs[graph_name].graph["completed_stages"]
            )

            actual_nodenames: Dict[str, int] = snn_graph.graph["results"]

            # Remove the passed boolean, and redo results verification.
            copy_actual_nodenames = copy.deepcopy(actual_nodenames)
            copy_actual_nodenames.pop("passed")

            # Verify node names are identical.
            test_object.assertEquals(
                copy_actual_nodenames.keys(), expected_nodenames.keys()
            )

            # Verify the expected nodes are the same as the actual nodes.
            for key, expected_val in expected_nodenames.items():
                test_object.assertEquals(
                    expected_val, copy_actual_nodenames[key]
                )

            test_object.assertTrue(actual_nodenames["passed"])
