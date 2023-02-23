"""Determines whether the snn algorithm is done."""

import networkx as nx
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked


@typechecked
def mdsa_is_done(
    *,
    run_config: Run_config,
    snn_graph: nx.DiGraph,
    t: int,
) -> bool:
    """Checks whether the network is done or not.

    First checks if terminator node is done. Then checks, if that is not
    the case whether radiation is active.
    """
    if list(run_config.algorithm.keys()) == ["MDSA"]:
        if snn_graph.nodes["terminator_node"]["nx_lif"][t].spikes:
            return True
        if run_config.radiation is not None:
            # Radiation may have killed any neuron. This may have arbitrarily
            # caused the neuron to not spike. This algorithm requires that
            # at least 1 selector neuron is firing within if t>1.
            for node_name in snn_graph.nodes:
                if a_neuron_is_spiking(
                    identifier="selector",
                    snn_graph=snn_graph,
                    t=t,
                ) or a_neuron_is_spiking(
                    identifier="next_round",
                    snn_graph=snn_graph,
                    t=t,
                ):
                    return False
                if "terminator" in node_name:
                    if (
                        snn_graph.nodes[node_name]["nx_lif"][t].spikes
                        and t > 0
                    ):
                        return True
            if t > 0:
                return True
            return False
        return False
    raise KeyError("Algorithm termination mode not yet found.")


def a_neuron_is_spiking(
    *, t: int, snn_graph: nx.DiGraph, identifier: str
) -> bool:
    """Returns True if a nextround neuron is spiking at timestep t."""
    for node_name in snn_graph.nodes:
        if identifier in node_name:
            if snn_graph.nodes[node_name]["nx_lif"][t].spikes and t > 0:
                return True
    return False
