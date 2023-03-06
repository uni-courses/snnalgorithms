"""Tests whether the `nth` redundant spike_once neuron in the MDSA algorithm
takes over from the died spike_once neurons (0 to n-1)."""
# pylint: disable=R0801
import copy
from pprint import pprint
from typing import Any, Dict, List, Union

import networkx as nx
from snncompare.exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.Plot_config import get_default_plot_config
from snncompare.optional_config.Output_config import Output_config
from snncompare.process_results.process_results import (
    compute_results,
    set_results,
)
from snncompare.run_config import Run_config
from snncompare.simulation.stage2_sim import sim_graphs
from snnradiation.Radiation_damage import verify_radiation_is_applied
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.get_results import get_results
from tests.sparse.MDSA.adaptation.helper import (
    assert_redundant_neuron_takes_over,
    create_default_output_config,
    long_exp_config_for_mdsa_testing_with_adaptation,
)
from tests.sparse.MDSA.test_snn_results import Test_mdsa_snn_results


# pylint: disable=R0903
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

        mdsa_settings: Exp_config = (
            long_exp_config_for_mdsa_testing_with_adaptation()
        )
        print("mdsa_settings=")
        pprint(mdsa_settings.__dict__)

        mdsa_settings.export_types = None
        output_config: Output_config = create_default_output_config(
            exp_config=mdsa_settings
        )
        full_exp_runner = Experiment_runner(
            exp_config=mdsa_settings,
            output_config=output_config,
            reverse=False,
            perform_run=False,
            specific_run_config=None,
        )

        for run_config in full_exp_runner.run_configs:
            print("run_config=")
            pprint(run_config.__dict__)
            if list(run_config.adaptation.keys()) == ["redundancy"]:
                exp_runner = Experiment_runner(
                    exp_config=mdsa_settings,
                    output_config=output_config,
                    reverse=False,
                    perform_run=False,
                    specific_run_config=run_config,
                )
                original_results_nx_graphs: Dict = (
                    exp_runner.perform_run_stage_1(
                        exp_config=mdsa_settings,
                        output_config=output_config,
                        plot_config=get_default_plot_config(),
                        run_config=run_config,
                    )
                )

                for dead_neuron_names in get_dead_neuron_name_cominations(
                    original_results_nx_graphs["graphs_dict"]["snn_algo_graph"]
                ):
                    if not any(
                        x in dead_neuron_names[0]
                        for x in ["counter", "terminator"]
                    ):
                        results_nx_graphs = copy.deepcopy(
                            original_results_nx_graphs
                        )
                        # Copy adapted graph into radiation graph to overwrite
                        # radiation death.
                        results_nx_graphs["graphs_dict"][
                            "rad_adapted_snn_graph"
                        ] = copy.deepcopy(
                            results_nx_graphs["graphs_dict"][
                                "adapted_snn_graph"
                            ]
                        )
                        rad_adapted_snn_graph = results_nx_graphs[
                            "graphs_dict"
                        ]["rad_adapted_snn_graph"]

                        # Set dead neuron names.

                        for dead_neuron_name in dead_neuron_names:
                            rad_adapted_snn_graph.nodes[dead_neuron_name][
                                "rad_death"
                            ] = True
                            rad_adapted_snn_graph.nodes[dead_neuron_name][
                                "nx_lif"
                            ][0].vth.set(9999)

                        verify_radiation_is_applied(
                            some_graph=rad_adapted_snn_graph,
                            dead_neuron_names=dead_neuron_names,
                            rad_type="neuron_death",
                        )

                        sim_graphs(
                            run_config=run_config,
                            stage_1_graphs=results_nx_graphs["graphs_dict"],
                        )

                        assert_redundant_neuron_takes_over(
                            dead_neuron_names=dead_neuron_names,
                            graphs_dict=results_nx_graphs["graphs_dict"],
                            output_config=output_config,
                            max_redundancy=list(
                                run_config.adaptation.values()
                            )[0],
                            run_config=run_config,
                            test_object=self,
                        )

                        # TODO: then also verify the complete adapted algorithm
                        # still works.
                        # perform_mdsa_results_computation_if_needed(
                        # m_val=run_config.algorithm["MDSA"]["m_val"],
                        # output_config=output_config,
                        # run_config=run_config,
                        # stage_2_graphs=results_nx_graphs["graphs_dict"],
                        # )

                        if set_results(
                            output_config=output_config,
                            run_config=run_config,
                            stage_2_graphs=results_nx_graphs["graphs_dict"],
                        ):
                            compute_results(
                                results_nx_graphs=results_nx_graphs,
                                stage_index=4,
                            )

                        # assert_run_config_json_results(
                        # test_object=self,
                        # graphs_dict=results_nx_graphs["graphs_dict"],
                        # run_config=run_config,
                        # )

            # TODO: run for radiation death of all combinations of
            # spike_once neurons (per node-cerciuit). E.g. n=0, n=1, n=0,1 etc.
            # self.assertTrue(True)
            # self.assertTrue(False)


@typechecked
def get_dead_neuron_name_cominations(
    snn_algo_graph: nx.DiGraph,
) -> List[List[str]]:
    """Returns dead neuron lists."""
    combinations: List[List[str]] = []
    # return [["selector_0_0"]]
    for node_name in snn_algo_graph.nodes():
        if "connector" not in node_name:
            combinations.append([node_name])

    # Also run a test where all original neurons have died.
    # combinations.append(list(snn_algo_graph.nodes()))

    return combinations


@typechecked
def assert_run_config_json_results(
    *,
    test_object: Any,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph]],
    run_config: Run_config,
) -> None:
    """Verifies the results of a run config using the json result output.

    TODO: update expected_stages.
    """

    # Verify results are as expected.
    expected_node_names: Dict[str, int] = get_results(
        input_graph=graphs_dict["input_graph"],
        m_val=run_config.algorithm["MDSA"]["m_val"],
        rand_props=graphs_dict["input_graph"].graph["alg_props"],
        seed=run_config.seed,
        size=run_config.graph_size,
    )

    for graph_name, snn_graph in graphs_dict.items():
        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:
            # Verify the SNN graphs have completed simulation stage 2.
            test_object.assertTrue(
                # pylint: disable=R1733
                2
                in graphs_dict[graph_name].graph["completed_stages"]
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
