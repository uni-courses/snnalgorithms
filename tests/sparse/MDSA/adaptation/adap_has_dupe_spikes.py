"""Checks whether an adaptated snn has duplicate spikes per neuron type in the
MDSA snn.

So whether multiple redundant neurons spike simultaneously per original
neuron.
"""
from typing import Dict, List, Union

import networkx as nx
from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked

from tests.sparse.MDSA.adaptation.redundancy_helper import (
    get_redundant_neuron_names,
    get_spike_window_per_neuron_type,
)


@typechecked
def assert_no_duplicate_spikes_in_adapted_network(
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph]],
    max_redundancy: int,
    run_config: Run_config,
    output_config: Output_config,
) -> None:
    """Raises exception if two spikes of the same neuron type are
    registered."""
    # For each spike in the adapted network, assert no redundant neuron spikes.
    adapted_graph: nx.DiGraph = graphs_dict["adapted_snn_graph"]
    snn_graph: nx.DiGraph = graphs_dict["snn_algo_graph"]

    # Per original node, check when it spikes in the adapted graph.
    for node_name in snn_graph:
        if not any(
            ignored_node_name in node_name
            for ignored_node_name in ["connector", "counter", "terminator"]
        ):
            for t, adapted_nx_lif in enumerate(
                adapted_graph.nodes[node_name]["nx_lif"]
            ):
                if adapted_nx_lif.spikes:
                    # Assert no duplicate spikes exist,
                    if redundant_neuron_spikes_at_original_t_spike_range(
                        adapted_graph=adapted_graph,
                        max_redundancy=max_redundancy,
                        node_name=node_name,
                        t=t,
                    ):
                        print(
                            "Error, redundant neuron spiked, while original "
                            + f"neuron did not spike, for: {node_name} at {t}."
                        )

                        # Visualise the snn behaviour

                        create_svg_plot(
                            graph_names=["adapted_snn_graph"],
                            graphs=graphs_dict,
                            output_config=output_config,
                            run_config=run_config,
                        )
                        raise ValueError()


@typechecked
def redundant_neuron_spikes_at_original_t_spike_range(
    *,
    adapted_graph: nx.DiGraph,
    max_redundancy: int,
    node_name: str,
    t: int,
) -> bool:
    """Returns True if a redundant neuron of a specific type spikes within the
    redundancy spike range of the original neuron.

    Returns False, the redundant neurons of a specific type do not spike
    within the redundancy spike range of the original neuron.
    """
    # For each spike in the adapted network, assert no redundant neuron spikes.
    timestep_window: List[int] = get_spike_window_per_neuron_type(
        t=t,
        max_redundancy=max_redundancy,
    )

    redundant_node_names: List[str] = get_redundant_neuron_names(
        max_redundancy=max_redundancy,
        original_node_name=node_name,
    )

    for redundant_node_name in redundant_node_names:
        for timestep in timestep_window:
            if (
                len(adapted_graph.nodes[redundant_node_name]["nx_lif"])
                > timestep
            ):
                if adapted_graph.nodes[redundant_node_name]["nx_lif"][
                    timestep
                ].spikes:
                    return True
            else:
                # The network is already done, so the redundant neurons cannot
                # spike anymore within the remaining timestep_window.
                break
    return False
