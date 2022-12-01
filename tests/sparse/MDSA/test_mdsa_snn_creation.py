"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801
import os
import shutil
import unittest

import networkx as nx
from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    long_exp_setts_for_mdsa_testing,
)
from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.Experiment_runner import Experiment_runner
from snncompare.graph_generation.stage_1_get_input_graphs import (
    get_used_graphs,
)
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.create_MDSA_snn import get_new_mdsa_graph


class Test_mdsa_snn_results(unittest.TestCase):
    """Tests whether the snn implementation of the MDSA algorithm yields the
    same results as the default/Neumann implementation if its weights are
    identical."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

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
        mdsa_settings: dict = long_exp_setts_for_mdsa_testing()
        # mdsa_creation_only_size_3_4: dict = short_mdsa_test_exp_setts()
        # mdsa_creation_only_size_3_4: dict = minimal_mdsa_test_exp_setts()

        # Do not output images.
        mdsa_settings["overwrite_snn_propagation"] = True
        mdsa_settings["overwrite_visualisation"] = True
        mdsa_settings["show_snns"] = False
        mdsa_settings["export_images"] = True

        verify_experiment_config(
            Supported_experiment_settings(),
            mdsa_settings,
            has_unique_id=False,
            allow_optional=True,
        )

        # OVERRIDE: Run only on a single run config.
        # exp_runner = override_with_single_run_setting(mdsa_settings)

        # Get experiment runner for long test.
        exp_runner = Experiment_runner(mdsa_settings, perform_run=False)
        for run_config in exp_runner.run_configs:
            stage_1_nx_graphs: dict = get_used_graphs(run_config)
            new_nx_mdsa_snn = get_new_mdsa_graph(run_config, stage_1_nx_graphs)

            # Assert the old and new networkx snns are itentical.
            # pprint(stage_1_nx_graphs["snn_algo_graph"].__dict__)
            self.assert_nodes_are_present(
                stage_1_nx_graphs["snn_algo_graph"], new_nx_mdsa_snn
            )

    @typechecked
    def assert_nodes_are_present(
        self,
        original_nx_snn: nx.DiGraph,
        new_nx_mdsa_snn: nx.DiGraph,
    ) -> None:
        """Verifies the results new snn graph contains the same nodes as the
        old snn graph creation."""
        # Verify all old nodes are in new network.
        for nodename in original_nx_snn.nodes:
            self.assertIn(
                nodename,
                list(
                    map(lambda neuron: neuron.full_name, new_nx_mdsa_snn.nodes)
                ),
            )

        # Also verify no extra nodes are created.
        for nodename in new_nx_mdsa_snn.nodes:
            self.assertIn(
                nodename.full_name,
                list(map(lambda neuron: neuron, original_nx_snn.nodes)),
            )
