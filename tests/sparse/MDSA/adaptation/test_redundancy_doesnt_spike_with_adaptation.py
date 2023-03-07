"""Tests redundant neurons don't spike when original neuron spikes.

Duration: 1h41
"""
# pylint: disable=R0801

import unittest
from pprint import pprint

from snncompare.simulation.stage2_sim import sim_graphs
from typeguard import typechecked

from tests.sparse.MDSA.adaptation.adap_has_dupe_spikes import (
    assert_no_duplicate_spikes_in_adapted_network,
)
from tests.sparse.MDSA.adaptation.redundancy_helper import (
    get_run_config_and_results_dicts_for_large_test_scope,
    overwrite_radiation_with_custom,
)


# pylint: disable=R0903
class Test_mdsa(unittest.TestCase):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_redundant_neurons_dont_spike_if_original_neuron_spikes(
        self,
    ) -> None:
        """Shows the network and raises an exception if a redundant neuron
        spikes within the time window of the spiking original neuron."""
        (
            run_config_results,
            output_config,
        ) = get_run_config_and_results_dicts_for_large_test_scope(
            with_adaptation_only=True
        )

        for i, (
            run_config,
            original_results_nx_graphs,
        ) in enumerate(run_config_results.items()):
            print(f"run_config ({i}/{len(run_config_results.keys())})=")
            pprint(run_config.__dict__)

            results_nx_graphs = overwrite_radiation_with_custom(
                original_results_nx_graphs=original_results_nx_graphs,
                dead_neuron_names=[],
            )

            # Now that the graphs have been created, simulate them.
            sim_graphs(
                run_config=run_config,
                stage_1_graphs=results_nx_graphs["graphs_dict"],
            )

            assert_no_duplicate_spikes_in_adapted_network(
                graphs_dict=results_nx_graphs["graphs_dict"],
                max_redundancy=list(run_config.adaptation.values())[0],
                run_config=run_config,
                output_config=output_config,
            )
