"""Creates the MDSA snn synapses."""


from typing import Dict

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron, Synapse
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.create_MDSA_snn_synapses import create_node_dict


@typechecked
def create_MDSA_recurrent_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    recurrent_weight: int,
    run_config: dict,
) -> nx.DiGraph:
    """Creates the synapses between the neurons for the MDSA algorithm."""

    node_dict = create_node_dict(mdsa_snn)

    # Create spike_once nodes.
    for node_index in input_graph.nodes:
        create_recurrent_spike_once_synapse(
            mdsa_snn, node_dict, node_index, recurrent_weight
        )

    for node_index in input_graph.nodes:
        for node_neighbour in nx.all_neighbors(input_graph, node_index):
            if node_index != node_neighbour:
                for m_val in range(
                    0, run_config["algorithm"]["MDSA"]["m_val"] + 1
                ):
                    create_recurrent_degree_receiver_synapse(
                        m_val,
                        mdsa_snn,
                        node_neighbour,
                        node_dict,
                        node_index,
                        recurrent_weight,
                    )

    # Create recurrent rand_synapses
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
            create_recurrent_rand_synapse(
                mdsa_snn,
                m_val,
                node_dict,
                node_index,
                recurrent_weight,
            )

    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        create_recurrent_next_round_synapse(
            mdsa_snn,
            m_val,
            node_dict,
        )
        create_recurrent_delay_synapse(
            mdsa_snn,
            m_val,
            node_dict,
        )

    return mdsa_snn


@typechecked
def create_recurrent_spike_once_synapse(
    mdsa_snn: nx.DiGraph,
    node_dict: Dict[str, LIF_neuron],
    node_index: int,
    recurrent_weight: int,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    mdsa_snn.add_edges_from(
        [
            (
                node_dict[f"spike_once_{node_index}"],
                node_dict[f"spike_once_{node_index}"],
            )
        ],
        weight=Synapse(
            weight=recurrent_weight,
            delay=0,
            change_per_t=0,
        ),
    )


# pylint: disable=R0913
@typechecked
def create_recurrent_degree_receiver_synapse(
    m_val: int,
    mdsa_snn: nx.DiGraph,
    neighbour_index: int,
    node_dict: Dict[str, LIF_neuron],
    node_index: int,
    recurrent_weight: int,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    mdsa_snn.add_edges_from(
        [
            (
                node_dict[
                    f"degree_receiver_{node_index}_{neighbour_index}_{m_val}"
                ],
                node_dict[
                    f"degree_receiver_{node_index}_{neighbour_index}_{m_val}"
                ],
            )
        ],
        weight=Synapse(
            weight=recurrent_weight,
            delay=0,
            change_per_t=0,
        ),
    )


@typechecked
def create_recurrent_rand_synapse(
    mdsa_snn: nx.DiGraph,
    m_val: int,
    node_dict: Dict[str, LIF_neuron],
    node_index: int,
    recurrent_weight: int,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    mdsa_snn.add_edges_from(
        [
            (
                node_dict[f"rand_{node_index}_{m_val}"],
                node_dict[f"rand_{node_index}_{m_val}"],
            )
        ],
        weight=Synapse(
            weight=recurrent_weight,
            delay=0,
            change_per_t=0,
        ),
    )


@typechecked
def create_recurrent_next_round_synapse(
    mdsa_snn: nx.DiGraph,
    m_val: int,
    node_dict: Dict[str, LIF_neuron],
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    mdsa_snn.add_edges_from(
        [
            (
                node_dict[f"next_round_{m_val}"],
                node_dict[f"next_round_{m_val}"],
            )
        ],
        weight=Synapse(
            weight=-5,  # TODO: why is this not "recurrent_weight"?
            delay=0,
            change_per_t=0,
        ),
    )


@typechecked
def create_recurrent_delay_synapse(
    mdsa_snn: nx.DiGraph,
    m_val: int,
    node_dict: Dict[str, LIF_neuron],
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    mdsa_snn.add_edges_from(
        [
            (
                node_dict[f"delay_{m_val}"],
                node_dict[f"delay_{m_val}"],
            )
        ],
        weight=Synapse(
            weight=-15,  # TODO: why is this not "recurrent_weight"?
            delay=0,
            change_per_t=0,
        ),
    )
