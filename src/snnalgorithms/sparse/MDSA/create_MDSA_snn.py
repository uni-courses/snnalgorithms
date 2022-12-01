"""Creates the MDSA snn."""


from pprint import pprint

import networkx as nx
from snnbackends.networkx.LIF_neuron import (
    Identifier,
    LIF_neuron,
    Recurrent_synapse,
)
from snncompare.helper import get_y_position
from typeguard import typechecked


@typechecked
def get_new_mdsa_graph(
    run_config: dict, stage_1_nx_graphs: dict
) -> nx.DiGraph:
    """Creates the networkx snn for a run configuration for the MDSA
    algorithm."""
    if not isinstance(
        stage_1_nx_graphs["input_graph"].graph["alg_props"], dict
    ):
        raise Exception("Error, algorithm properties not set.")

    return create_MDSA_neurons(run_config, stage_1_nx_graphs["input_graph"])


@typechecked
def create_MDSA_neurons(run_config: dict, input_graph: nx.Graph) -> nx.DiGraph:
    """Creates the neurons for the MDSA algorithm."""
    mdsa_snn = nx.DiGraph()
    # TODO get spacing from somewhere.
    spacing = 0.25
    recurrent_weight = -10  # TODO: explain why 10.

    # Create connecting node.
    connecting_nodes = create_connecting_node(spacing)
    mdsa_snn.add_node(connecting_nodes)

    # Create spike_once nodes.
    for node_index in input_graph.nodes:
        spike_once_node = create_spike_once_node(
            spacing, node_index, recurrent_weight
        )
        mdsa_snn.add_node(spike_once_node)

    # Create degree_receiver nodes.
    for node_index in input_graph.nodes:
        for node_neighbour in nx.all_neighbors(input_graph, node_index):
            if node_index != node_neighbour:
                for m_val in range(
                    0, run_config["algorithm"]["MDSA"]["m_val"] + 1
                ):
                    pprint(run_config)
                    degree_receiver_node = create_degree_receiver_node(
                        input_graph,
                        m_val,
                        node_index,
                        node_neighbour,
                        recurrent_weight,
                        spacing,
                    )
                    mdsa_snn.add_node(degree_receiver_node)

    # Create random spike nodes.
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
            rand_node = create_rand_node(
                m_val,
                node_index,
                spacing,
                recurrent_weight,
            )
            mdsa_snn.add_node(rand_node)

    return mdsa_snn


@typechecked
def create_connecting_node(spacing: float) -> LIF_neuron:
    """Creates the neuron settings for the connecting node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="connecting_node",
        bias=0.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(float(-spacing), float(spacing)),
    )


def create_spike_once_node(
    spacing: float, node_index: int, recurrent_weight: int
) -> LIF_neuron:
    """Creates the neuron settings for the spike_once node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="spike_once",
        bias=2.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(float(0), float(node_index * 4 * spacing)),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index)
        ],
        recurrent_synapses=[
            Recurrent_synapse(
                weight=recurrent_weight,
                delay=0,
                change_per_t=0,
            )
        ],
    )


# pylint: disable=R0913
def create_degree_receiver_node(
    input_graph: nx.Graph,
    m_val: int,
    node_index: int,
    node_neighbour: int,
    recurrent_weight: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the spike_once node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="degree_receiver",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=1.0,
        pos=(
            float(4 * spacing + m_val * 9 * spacing),
            get_y_position(input_graph, node_index, node_neighbour, spacing),
        ),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index),
            Identifier(
                description="neighbour_index", position=1, value=node_neighbour
            ),
            Identifier(description="m_val", position=2, value=m_val),
        ],
        recurrent_synapses=[
            Recurrent_synapse(
                weight=recurrent_weight,
                delay=0,
                change_per_t=0,
            )
        ],
    )


def create_rand_node(
    m_val: int,
    node_index: int,
    spacing: float,
    recurrent_weight: int,
) -> LIF_neuron:
    """Creates the neuron settings for the rand node in the MDSA algorithm."""
    return LIF_neuron(
        name="rand",
        bias=2.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(
            float(spacing + m_val * 9 * spacing),
            float(node_index * 4 * spacing) + spacing,
        ),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index),
            # TODO: remove m_val dependency.
            Identifier(description="m_val", position=1, value=m_val),
        ],
        recurrent_synapses=[
            Recurrent_synapse(
                weight=recurrent_weight,
                delay=0,
                change_per_t=0,
            )
        ],
    )
