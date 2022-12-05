"""Tests whether the old snn MDSA algorithm neurons equal new/simplified
neurons.

This is moved from the test_mdsa_snn_creation function, because testing
the nodes, edges and synapse values do not require the run config to be
propagated, whereas comparing the LIF neuron values, currently does
require so. This way, the other 3 tests can still be executed fast, and
this slow test is separated.
"""
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

from snnalgorithms.sparse.MDSA.create_MDSA_snn_neurons import (
    get_new_mdsa_graph,
)
from snnalgorithms.sparse.MDSA.create_MDSA_snn_recurrent_synapses import (
    create_MDSA_recurrent_synapses,
)
from snnalgorithms.sparse.MDSA.create_MDSA_snn_synapses import create_node_dict


class Test_mdsa_snn_results(unittest.TestCase):
    """Tests whether the snn implementation of the MDSA algorithm yields the
    same results as the default/Neumann implementation if its weights are
    identical."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_snn_results_equal_neumann_results(self) -> None:
        """Tests whether the results of the snn implementation of the MDSA
        algorithm are the same as those of the default/Neumann implementation
        of that MDSA algorithm. ."""

        # TODO: move to central place in MDSA algo spec.
        recurrent_weight = -10

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
        mdsa_settings["overwrite_visualisation"] = False
        mdsa_settings["show_snns"] = False
        mdsa_settings["export_images"] = False

        verify_experiment_config(
            Supported_experiment_settings(),
            mdsa_settings,
            has_unique_id=False,
            allow_optional=True,
        )

        # OVERRIDE: Run only on a single run config.
        # exp_runner = override_with_single_run_setting(mdsa_settings)

        # TODO: run only for a single run config at a time, then perform
        # asserts, then move to next runconfig. Do this for all run configs.

        # Get experiment runner for long test.
        full_exp_runner = Experiment_runner(
            mdsa_settings,
            perform_run=False,
        )
        for run_config in full_exp_runner.run_configs:
            exp_runner = Experiment_runner(
                mdsa_settings,
                specific_run_config=run_config,
                perform_run=True,
            )

            # Used to get the input graph.
            stage_1_nx_graphs: dict = get_used_graphs(run_config)
            new_nx_mdsa_snn = get_new_mdsa_graph(run_config, stage_1_nx_graphs)

            create_MDSA_recurrent_synapses(
                stage_1_nx_graphs["input_graph"],
                new_nx_mdsa_snn,
                recurrent_weight,
                run_config,
            )

            self.assert_neuron_values_are_identical(
                exp_runner,
                new_nx_mdsa_snn,
                run_config,
            )

    @typechecked
    def assert_neuron_values_are_identical(
        self,
        exp_runner: Experiment_runner,
        new_nx_mdsa_snn: nx.DiGraph,
        run_config: dict,
    ) -> None:
        """Verifies the results new snn graph contains the same neuron values
        as the old snn graph creation."""

        # Verify all old nodes are in new network.
        node_dict = create_node_dict(new_nx_mdsa_snn)

        # Get propagated graphs.
        propagated_original_nx_snn: nx.DiGraph = exp_runner.results_nx_graphs[
            run_config["unique_id"]
        ]["graphs_dict"]["snn_algo_graph"]

        # Get LIF neurons
        for nodename in propagated_original_nx_snn.nodes:
            # print(propagated_original_nx_snn.nodes[nodename]["nx_LIF"])
            timestep = (
                0  # Get the first initialised LIF neuron from simulation.
            )
            original_lif = propagated_original_nx_snn.nodes[nodename][
                "nx_LIF"
            ][timestep]

            # Compare LIF neuron values.
            new_lif = node_dict[nodename]
            print(f"nodename={nodename}")

            self.assertEqual(original_lif.bias.bias, new_lif.bias.bias)
            self.assertEqual(original_lif.du.du, new_lif.du.du)
            self.assertEqual(original_lif.dv.dv, new_lif.dv.dv)
            self.assertEqual(original_lif.vth.vth, new_lif.vth.vth)

            # Du
            # Dv
            # Bias
            # Vth
            # V_reset
