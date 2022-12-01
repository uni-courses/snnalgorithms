"""Creates the MDSA snn neurons."""


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


# pylint: disable=R0912
# pylint: disable=R0914
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

    # Create selector nodes.
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
            selector_node = create_selector_node(
                m_val,
                node_index,
                spacing,
            )
            mdsa_snn.add_node(selector_node)

    # Create selector nodes.
    for node_index in input_graph.nodes:
        counter_node = create_counter_node(
            run_config["algorithm"]["MDSA"]["m_val"],
            node_index,
            spacing,
        )
        mdsa_snn.add_node(counter_node)

    # NOTE, for loop starts at index 1, instead of 0!
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        next_round_node = create_next_round_node(
            m_val,
            len(input_graph.nodes),
            spacing,
        )
        mdsa_snn.add_node(next_round_node)

    # NOTE, for loop starts at index 1, instead of 0!
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        d_charger = create_d_charger_node(
            m_val,
            spacing,
        )
        mdsa_snn.add_node(d_charger)

    # NOTE, for loop starts at index 1, instead of 0!
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        delay = create_delay_node(
            m_val,
            len(input_graph.nodes),
            spacing,
        )
        mdsa_snn.add_node(delay)

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


def create_selector_node(
    m_val: int,
    node_index: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the selector node in the MDSA
    algorithm."""
    # TODO: why. This is probably for the delay in activation for m>0.
    bias: float
    if m_val == 0:
        bias = 4.0
    else:
        bias = 5.0

    return LIF_neuron(
        name="selector",
        bias=bias,
        du=0.0,
        dv=1.0,
        vth=4.0,
        pos=(
            float(7 * spacing + m_val * 9 * spacing),
            float(node_index * 4 * spacing + spacing),
        ),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index),
            Identifier(description="m_val", position=1, value=m_val),
        ],
    )


def create_counter_node(
    m_val: int,
    node_index: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the counter node in the MDSA
    algorithm."""

    return LIF_neuron(
        name="counter",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=0.0,
        pos=(
            float(9 * spacing + m_val * 9 * spacing),
            float(node_index * 4 * spacing),
        ),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index),
            # TODO: remove m_val dependency.
            Identifier(description="m_val", position=1, value=m_val),
        ],
    )


def create_next_round_node(
    m_val: int,
    nr_of_nodes: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the counter node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="next_round",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=float(nr_of_nodes),
        pos=(float(6 * spacing + (m_val - 1) * 9 * spacing), -2 * spacing),
        identifiers=[
            Identifier(description="m_val", position=0, value=m_val),
        ],
    )


def create_d_charger_node(
    m_val: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the d_charger node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="d_charger",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=0.0,
        pos=(float(9 * spacing + (m_val - 1) * 9 * spacing), -2 * spacing),
        identifiers=[
            Identifier(description="m_val", position=0, value=m_val),
        ],
    )


def create_delay_node(
    m_val: int,
    nr_of_nodes: int,
    spacing: float,
) -> LIF_neuron:
    """Creates the neuron settings for the delay node in the MDSA algorithm."""
    return LIF_neuron(
        name="delay",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=float(2 * nr_of_nodes - 1),
        pos=(float(12 * spacing + (m_val - 1) * 9 * spacing), -2 * spacing),
        identifiers=[
            Identifier(description="m_val", position=0, value=m_val),
        ],
    )
