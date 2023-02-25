"""Tests whether the `nth` redundant spike_once neuron in the MDSA algorithm
takes over from the died spike_once neurons (0 to n-1)."""
# pylint: disable=R0801
import copy
from pprint import pprint
from typing import Dict, List

from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.Plot_config import get_default_plot_config
from snncompare.optional_config.Output_config import (
    Extra_storing_config,
    Output_config,
    Zoom,
)
from snnradiation.Radiation_damage import verify_radiation_is_applied
from typeguard import typechecked

from tests.sparse.MDSA.test_snn_results import Test_mdsa_snn_results


class Test_mdsa(Test_mdsa_snn_results):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super(Test_mdsa_snn_results, self).__init__(*args, **kwargs)
        # Generate default experiment config.
        self.create_exp_config()

    @typechecked
    def test_something(self) -> None:
        """Tests whether the MDSA algorithm with adaptation yields the same
        results as without adaptation."""
        # TODO: load all MDSA configs.
        for redundancy in range(2, 6, 2):
            # Modify configuration to include adaptation.
            self.mdsa_settings.adaptations = {"redundancy": [redundancy]}
            self.mdsa_settings.radiations = {"neuron_death": [0.25]}

            # Narrow down test scope by overriding experiment settings.
            # self.mdsa_settings.size_and_max_graphs = [(4, 1)]
            self.mdsa_settings.algorithms = {
                "MDSA": [
                    {"m_val": 1},
                ]
            }
            self.mdsa_settings.export_types = None
            self.mdsa_settings.size_and_max_graphs = [(3, 1)]
            # pprint(self.mdsa_settings.__dict__)

            # TODO: get a MDSA snn that has a redundancy of n =2
            # TODO: get a MDSA snn that has a redundancy of n =5
            # ...

            # TODO: find random seed for MDSA snn for n = 2 that makes the
            # spike_once neuron with redundancies [(0), (0,1)] die,

            # Create duplicate Output_config that is used to generate the data
            # belonging to each run config, using the Experiment runner.
            output_config = Output_config(
                recreate_stages=[],
                export_types=[],
                zoom=Zoom(
                    create_zoomed_image=False,
                    left_right=None,
                    bottom_top=None,
                ),
                output_json_stages=[1, 2, 4],
                extra_storing_config=Extra_storing_config(
                    count_spikes=False,
                    count_neurons=False,
                    count_synapses=False,
                    show_images=False,
                    store_died_neurons=False,
                ),
            )

            # Get experiment runner for long test.
            full_exp_runner = Experiment_runner(
                exp_config=self.mdsa_settings,
                output_config=output_config,
                reverse=False,
                perform_run=False,
                specific_run_config=None,
            )
            for run_config in full_exp_runner.run_configs:
                exp_runner = Experiment_runner(
                    exp_config=self.mdsa_settings,
                    output_config=output_config,
                    reverse=False,
                    perform_run=False,
                    specific_run_config=run_config,
                )
                print("Generate graphs in stage 1")
                results_nx_graphs: Dict = exp_runner.perform_run_stage_1(
                    exp_config=self.mdsa_settings,
                    output_config=output_config,
                    plot_config=get_default_plot_config(),
                    run_config=run_config,
                )

                # print("results_nx_graphs")
                pprint(results_nx_graphs["graphs_dict"])

                results_nx_graphs["graphs_dict"][
                    "rad_adapted_snn_graph"
                ] = copy.deepcopy(
                    results_nx_graphs["graphs_dict"]["adapted_snn_graph"]
                )
                rad_adapted_snn_graph = results_nx_graphs["graphs_dict"][
                    "rad_adapted_snn_graph"
                ]
                # print(f'adapted_snn_graph={adapted_snn_graph}')

                # Set dead neuron name
                print(f"rad_adapted_snn_graph={rad_adapted_snn_graph}")
                # for node_name in rad_adapted_snn_graph.nodes():
                dead_neuron_names: List[str] = ["spike_once_0"]
                for dead_neuron_name in dead_neuron_names:
                    rad_adapted_snn_graph.nodes[dead_neuron_name][
                        "rad_death"
                    ] = True
                    rad_adapted_snn_graph.nodes[dead_neuron_name]["nx_lif"][
                        0
                    ].vth.set(9999)

                verify_radiation_is_applied(
                    some_graph=rad_adapted_snn_graph,
                    dead_neuron_names=dead_neuron_names,
                    rad_type="neuron_death",
                )

                # TODO: either simulate 1 timestep at a time, or simulate
                # with stage_2.

                # TODO  Verify the redundant spike_once neuron of [(1),(2)]
                # takes over.

            # TODO: then also verify the complete adapted algorithm still
            # works.

            # TODO: run for radiation death of all combinations of
            # spike_once neurons (per node-cerciuit). E.g. n=0, n=1, n=0,1 etc.
            # self.assertTrue(True)
            # self.assertTrue(False)
