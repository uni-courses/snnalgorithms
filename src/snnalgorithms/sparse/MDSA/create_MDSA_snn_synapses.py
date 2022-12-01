"""Creates the MDSA snn synapses."""


from typing import Dict

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked


def create_node_dict(mdsa_snn: nx.DiGraph) -> Dict[str, LIF_neuron]:
    """Creates a dictionary with full_name of the node as key, and the actual
    LIF_neuron object as value."""
    node_dict: Dict[str, LIF_neuron] = {}
    for lif_neuron in mdsa_snn.nodes:
        # pylint: disable=C0201
        if lif_neuron.full_name not in node_dict.keys():
            node_dict[lif_neuron.full_name] = lif_neuron
        else:
            raise Exception(
                "No duplicate nodenames permitted in"
                + f" snn:{lif_neuron.full_name}"
            )
    return node_dict


@typechecked
def create_MDSA_synapses(
    input_graph: nx.Graph, mdsa_snn: nx.DiGraph
) -> nx.DiGraph:
    """Creates the synapses between the neurons for the MDSA algorithm."""

    node_dict = create_node_dict(mdsa_snn)

    # Create synapses for connecting node.
    create_connecting_node_synapses(input_graph, mdsa_snn, node_dict)

    # Create spike_once nodes.
    for _ in input_graph.nodes:
        create_connecting_node_synapses(input_graph, mdsa_snn, node_dict)

    return mdsa_snn


@typechecked
def create_connecting_node_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    node_dict: Dict[str, LIF_neuron],
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    for node_index in input_graph.nodes:
        mdsa_snn.add_edges_from(
            [
                (
                    node_dict["connecting_node"],
                    node_dict[f"spike_once_{node_index}"],
                )
            ],
            weight=0,
        )


@typechecked
def create_connecting_spike_once_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    node_dict: Dict[str, LIF_neuron],
) -> None:
    """Creates the outgoing synapses for the spike_once node in the MDSA
    algorithm."""
    for node_index in input_graph.nodes:
        mdsa_snn.add_edges_from(
            [
                (
                    node_dict[f"spike_once_{node_index}"],
                    node_dict["connecting_node"],
                )
            ],
            weight=0,
        )
