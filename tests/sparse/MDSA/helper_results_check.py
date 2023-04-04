"""Helps verify the snn graph results using the Neumann algorithm results."""
import copy
from typing import Any, Dict, Optional, Union

import networkx as nx
from snncompare.export_results.load_json_to_nx_graph import (
    load_json_to_nx_graph_from_file,
)
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.get_results import get_neumann_results


@typechecked
def assert_run_config_results_without_rad(
    *,
    from_json: bool,
    test_object: Any,
    run_config: Run_config,
    nx_graphs: Optional[Dict[str, Union[nx.Graph, nx.DiGraph]]] = None,
) -> None:
    """Verifies the results of the original snn graph and the adapted snn graph
    without radiation, against the Neumann MDSA algorithm.

    if from_json is True: loads the results from json json file for
    verification. if from_json is False: verifies results in the
    nx_graphs dict.
    """
    if from_json:
        nx_graphs = load_json_to_nx_graph_from_file(
            run_config=run_config, stage_index=4, expected_stages=[1, 2, 3, 4]
        )
    elif nx_graphs is None:
        raise ValueError("Error, did not find nx_graphs to test.")
    assert nx_graphs is not None  # nosec

    # Verify results are as expected.
    expected_node_names: Dict[str, int] = get_neumann_results(
        input_graph=nx_graphs["input_graph"],
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
            test_object.assertEqual(
                copy_actual_node_names.keys(), expected_node_names.keys()
            )

            # Verify the expected nodes are the same as the actual nodes.
            for key, expected_val in expected_node_names.items():
                test_object.assertEqual(
                    expected_val, copy_actual_node_names[key]
                )

            test_object.assertTrue(actual_node_names["passed"])
