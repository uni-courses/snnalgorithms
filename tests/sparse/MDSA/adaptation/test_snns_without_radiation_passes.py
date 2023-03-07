"""Tests whether the adaptation graphs have valid results. Also tests whether
the default snn_algo_graph has valid results.

TODO: determine whether it asserts the results are the same as the original
snn, or as the Neumann algo results.
TODO: this is a duplicate test remove the original.
TODO: verify if `set_results(` performs assert of result validity.
TODO: include the actual assert, the compute_results( does not
assert valid results.
"""
# pylint: disable=R0801
import unittest
from pprint import pprint

from snncompare.process_results.process_results import (
    compute_results,
    set_results,
)
from snncompare.simulation.stage2_sim import sim_graphs
from typeguard import typechecked

from tests.sparse.MDSA.adaptation.redundancy_helper import (
    get_run_config_and_results_dicts_for_large_test_scope,
    overwrite_radiation_with_custom,
)
from tests.sparse.MDSA.helper_results_check import (
    assert_run_config_results_without_rad,
)


# pylint: disable=R0903
class Test_mdsa(unittest.TestCase):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_something(self) -> None:
        """Tests whether the MDSA algorithm with adaptation yields the same
        results as without adaptation."""

        # (Currently), the graphs with adaptation, also contain the snns
        # without adaptation, so to prevent duplicate evaluation, of the snns
        # without adaptation, ask only with adaptation.
        (
            run_config_results,
            output_config,
        ) = get_run_config_and_results_dicts_for_large_test_scope(
            with_adaptation_only=True
        )

        for (
            run_config,
            original_results_nx_graphs,
        ) in run_config_results.items():
            print("run_config=")
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

            assert_run_config_results_without_rad(
                from_json=False,
                test_object=self,
                run_config=run_config,
                nx_graphs=results_nx_graphs["graphs_dict"],
            )
