"""Tests whether the `nth` redundant neuron (with n=1) in the MDSA algorithm
takes over from the died neurons (0 to n-1)."""
# pylint: disable=R0801

from pprint import pprint

from snncompare.process_results.process_results import (
    compute_results,
    set_results,
)
from snncompare.simulation.stage2_sim import sim_graphs
from typeguard import typechecked

from tests.sparse.MDSA.adaptation.redundancy_helper import (
    assert_redundant_neuron_takes_over,
    get_dead_neuron_name_cominations,
    get_run_config_and_results_dicts_for_large_test_scope,
    overwrite_radiation_with_custom,
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
        (
            run_config_results,
            output_config,
        ) = get_run_config_and_results_dicts_for_large_test_scope()

        for (
            run_config,
            original_results_nx_graphs,
        ) in run_config_results.items():
            print("run_config=")
            pprint(run_config.__dict__)

            # Generate lists with dead neurons that are to be considered
            # dead during a run. One list contains all the neurons that
            # will be dead due to radiation in a single run_config.
            for dead_neuron_names in get_dead_neuron_name_cominations(
                snn_algo_graph=original_results_nx_graphs["graphs_dict"][
                    "snn_algo_graph"
                ]
            ):
                # Do not test redundancy for counter neuron, because they
                # don't spike.
                # Do not test redundancy for terminator node because it
                # is ok if any of them fire simultaneously, so they are
                # "dumb" copies, not intelligent redundancy.
                if not any(
                    x in dead_neuron_names[0]
                    for x in ["counter", "terminator"]
                ):
                    results_nx_graphs = overwrite_radiation_with_custom(
                        original_results_nx_graphs=original_results_nx_graphs,
                        dead_neuron_names=dead_neuron_names,
                    )

                    # Now that the graphs have been created, simulate them.
                    sim_graphs(
                        run_config=run_config,
                        stage_1_graphs=results_nx_graphs["graphs_dict"],
                    )

                    # Perform actual test.
                    assert_redundant_neuron_takes_over(
                        dead_neuron_names=dead_neuron_names,
                        graphs_dict=results_nx_graphs["graphs_dict"],
                        output_config=output_config,
                        max_redundancy=list(run_config.adaptation.values())[0],
                        run_config=run_config,
                        test_object=self,
                    )

                    # Then also verify the complete adapted algorithm
                    # still works.
                    if set_results(
                        output_config=output_config,
                        run_config=run_config,
                        stage_2_graphs=results_nx_graphs["graphs_dict"],
                    ):
                        compute_results(
                            results_nx_graphs=results_nx_graphs,
                            stage_index=4,
                        )
