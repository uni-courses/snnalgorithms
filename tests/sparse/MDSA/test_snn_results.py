"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801
import copy
import os
import shutil
import unittest
from pprint import pprint
from typing import TYPE_CHECKING, Any, Dict

from snncompare.exp_config.Exp_config import (
    Supported_experiment_settings,
    verify_exp_config,
)
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_results.load_json_to_nx_graph import (
    load_json_to_nx_graph_from_file,
)
from snncompare.json_configurations.run_configs.algo_test import (
    long_exp_config_for_mdsa_testing,
    run_config_with_error,
)
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnalgorithms.sparse.MDSA.get_results import get_results

if TYPE_CHECKING:
    from snncompare.exp_config.Exp_config import Exp_config


class Test_mdsa_snn_results(unittest.TestCase):
    """Tests whether the snn implementation of the MDSA algorithm yields the
    same results as the default/Neumann implementation if its weights are
    identical."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def create_exp_config(self) -> None:
        """Generates the default test settings for the MDSA SNN
        implementations."""
        # Generate default experiment config.
        # pylint: disable=W0201
        self.mdsa_settings: Exp_config = long_exp_config_for_mdsa_testing()

        # Do not output images.
        self.mdsa_settings.recreate_s2 = True
        self.mdsa_settings.overwrite_images_only = False
        self.mdsa_settings.export_images = False
        self.mdsa_settings.export_types = ["png"]

    @typechecked
    def helper(self, mdsa_settings: Dict) -> None:
        """Tests whether the results of the snn implementation of the MDSA
        algorithm are the same as those of the default/Neumann implementation
        of that MDSA algorithm. ."""
        pprint(mdsa_settings)

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("latex"):
            shutil.rmtree("latex")

        verify_exp_config(
            supp_exp_config=Supported_experiment_settings(),
            exp_config=mdsa_settings,
            has_unique_id=False,
            allow_optional=True,
        )

        # OVERRIDE: Run only on a single run config.
        # exp_runner = override_with_single_run_setting(mdsa_settings)

        # Get experiment runner for long test.
        full_exp_runner = Experiment_runner(
            mdsa_settings,
            perform_run=False,
        )
        for run_config in full_exp_runner.run_configs:
            Experiment_runner(
                mdsa_settings,
                specific_run_config=run_config,
                perform_run=True,
            )

            # Verify results are identical using the json results file.
            assert_run_config_json_results(
                test_object=self, run_config=run_config
            )


@typechecked
def override_with_single_run_setting(
    *,
    mdsa_settings: Exp_config,
) -> Experiment_runner:
    """Overwrites a list of experiment settings to only run the experiment on a
    single run configuration."""
    algorithms = {
        "MDSA": get_algo_configs(algo_spec=MDSA(list(range(0, 1, 1))).__dict__)
    }
    mdsa_settings.algorithms = algorithms
    some_run_config_with_error = run_config_with_error()
    some_run_config_with_error.export_images = True
    exp_runner = Experiment_runner(mdsa_settings, some_run_config_with_error)
    return exp_runner


@typechecked
def assert_run_config_json_results(
    *,
    test_object: Any,
    run_config: Run_config,
) -> None:
    """Verifies the results of a run config using the json result output.

    TODO: update expected_stages.
    """

    nx_graphs = load_json_to_nx_graph_from_file(
        run_config=run_config, stage_index=4, expected_stages=[1, 2, 3, 4]
    )

    # Verify results are as expected.
    expected_node_names: Dict[str, int] = get_results(
        input_graph=nx_graphs["input_graph"],
        iteration=run_config.iteration,
        m_val=run_config.algorithm["MDSA"]["m_val"],
        rand_props=nx_graphs["input_graph"].graph["alg_props"],
        seed=run_config.seed,
        size=run_config.graph_size,
    )

    for graph_name, snn_graph in nx_graphs.items():

        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:

            # Verify the SNN graphs have completed simulation stage 2.
            test_object.assertTrue(
                # pylint: disable=R1733
                2
                in nx_graphs[graph_name].graph["completed_stages"]
            )

            actual_node_names: Dict[str, int] = snn_graph.graph["results"]

            # Remove the passed boolean, and redo results verification.
            copy_actual_node_names = copy.deepcopy(actual_node_names)
            copy_actual_node_names.pop("passed")

            # Verify node names are identical.
            test_object.assertEquals(
                copy_actual_node_names.keys(), expected_node_names.keys()
            )

            # Verify the expected nodes are the same as the actual nodes.
            for key, expected_val in expected_node_names.items():
                test_object.assertEquals(
                    expected_val, copy_actual_node_names[key]
                )

            test_object.assertTrue(actual_node_names["passed"])
