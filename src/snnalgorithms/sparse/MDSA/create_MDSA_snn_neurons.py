"""Creates the MDSA snn neurons."""


from typing import Dict

import networkx as nx
from snnbackends.networkx.LIF_neuron import Identifier, LIF_neuron
from snncompare.helper import get_y_position
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.create_MDSA_snn_recurrent_synapses import (
    create_MDSA_recurrent_synapses,
)
from snnalgorithms.sparse.MDSA.create_MDSA_snn_synapses import (
    create_MDSA_synapses,
)


@typechecked
def get_new_mdsa_graph(run_config: dict, input_graph: nx.Graph) -> nx.DiGraph:
    """Creates the networkx snn for a run configuration for the MDSA
    algorithm."""
    if not isinstance(input_graph.graph["alg_props"], dict):
        raise Exception("Error, algorithm properties not set.")

    # TODO get spacing and recurrent weight form algo specification.
    spacing: float = 0.25
    recurrent_weight: int = -10

    snn_graph = create_MDSA_neurons(input_graph, run_config, spacing)

    create_MDSA_recurrent_synapses(
        input_graph,
        snn_graph,
        recurrent_weight,
        run_config,
    )

    create_MDSA_synapses(
        input_graph,
        snn_graph,
        run_config,
    )

    return snn_graph


# pylint: disable=R0912
# pylint: disable=R0914
@typechecked
def create_MDSA_neurons(
    input_graph: nx.Graph, run_config: dict, spacing: float
) -> nx.DiGraph:
    """Creates the neurons for the MDSA algorithm."""
    mdsa_snn = nx.DiGraph()

    # Create connecting node.
    create_connecting_node(mdsa_snn, spacing)

    # Create spike_once nodes.
    create_spike_once_node(input_graph, mdsa_snn, spacing)

    create_degree_receiver_node(
        input_graph,
        mdsa_snn,
        run_config,
        spacing,
    )

    # Create random spike nodes.
    create_rand_node(
        input_graph,
        mdsa_snn,
        run_config,
        spacing,
    )

    # Create selector nodes.
    create_selector_node(
        input_graph,
        mdsa_snn,
        run_config,
        spacing,
    )

    # Create selector nodes.
    create_counter_node(
        input_graph,
        run_config["algorithm"]["MDSA"]["m_val"],
        mdsa_snn,
        spacing,
    )

    create_next_round_node(
        mdsa_snn,
        len(input_graph.nodes),
        run_config,
        spacing,
    )

    create_terminator_node(
        input_graph,
        run_config["algorithm"]["MDSA"]["m_val"],
        mdsa_snn,
        spacing,
    )

    return mdsa_snn


@typechecked
def create_connecting_node(mdsa_snn: nx.DiGraph, spacing: float) -> None:
    """Creates the neuron settings for the connecting node in the MDSA
    algorithm."""
    lif_neuron = LIF_neuron(
        name="connecting_node",
        bias=0.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(float(-spacing), float(spacing)),
    )
    mdsa_snn.add_node(lif_neuron.full_name)
    mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


@typechecked
def create_spike_once_node(
    input_graph: nx.Graph, mdsa_snn: nx.DiGraph, spacing: float
) -> None:
    """Creates the neuron settings for the spike_once node in the MDSA
    algorithm."""
    for node_index in input_graph.nodes:
        lif_neuron = LIF_neuron(
            name="spike_once",
            bias=2.0,
            du=0.0,
            dv=0.0,
            vth=1.0,
            pos=(float(0), float(node_index * 4 * spacing)),
            identifiers=[
                Identifier(
                    description="node_index",
                    position=0,
                    value=node_index,
                )
            ],
        )

        mdsa_snn.add_node(lif_neuron.full_name)
        mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


# pylint: disable=R0913
@typechecked
def create_degree_receiver_node(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
    spacing: float,
) -> None:
    """Creates the neuron settings for the spike_once node in the MDSA
    algorithm."""
    # pylint: disable=R0801
    # Create degree_receiver nodes.
    for node_index in input_graph.nodes:
        for node_neighbour in nx.all_neighbors(input_graph, node_index):
            if node_index != node_neighbour:
                for m_val in range(
                    0, run_config["algorithm"]["MDSA"]["m_val"] + 1
                ):
                    lif_neuron = LIF_neuron(
                        name="degree_receiver",
                        bias=0.0,
                        du=0.0,
                        dv=1.0,
                        vth=1.0,
                        pos=(
                            float(4 * spacing + m_val * 9 * spacing),
                            get_y_position(
                                input_graph,
                                node_index,
                                node_neighbour,
                                spacing,
                            ),
                        ),
                        identifiers=[
                            Identifier(
                                description="node_index",
                                position=0,
                                value=node_index,
                            ),
                            Identifier(
                                description="neighbour_index",
                                position=1,
                                value=node_neighbour,
                            ),
                            Identifier(
                                description="m_val",
                                position=2,
                                value=m_val,
                            ),
                        ],
                    )

                    mdsa_snn.add_node(lif_neuron.full_name)
                    mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [
                        lif_neuron
                    ]


@typechecked
def create_rand_node(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
    spacing: float,
) -> None:
    """Creates the neuron settings for the rand node in the MDSA algorithm."""
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
            lif_neuron = LIF_neuron(
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
                    Identifier(
                        description="node_index",
                        position=0,
                        value=node_index,
                    ),
                    # TODO: remove m_val dependency.
                    Identifier(description="m_val", position=1, value=m_val),
                ],
            )
            mdsa_snn.add_node(lif_neuron.full_name)
            mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


@typechecked
def create_selector_node(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
    spacing: float,
) -> None:
    """Creates the neuron settings for the selector node in the MDSA
    algorithm."""
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
            # TODO: why. This is probably for the delay in activation for m>0.
            bias: float
            if m_val == 0:
                bias = 5.0
            else:
                bias = 4.0  # read this wrong looked like it was 4
                # in original code.
            lif_neuron = LIF_neuron(
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
                    Identifier(
                        description="node_index",
                        position=0,
                        value=node_index,
                    ),
                    Identifier(description="m_val", position=1, value=m_val),
                ],
            )
            mdsa_snn.add_node(lif_neuron.full_name)
            mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


@typechecked
def create_counter_node(
    input_graph: nx.Graph,
    m_val: int,
    mdsa_snn: nx.DiGraph,
    spacing: float,
) -> None:
    """Creates the neuron settings for the counter node in the MDSA
    algorithm."""
    for node_index in input_graph.nodes:
        lif_neuron = LIF_neuron(
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
                Identifier(
                    description="node_index",
                    position=0,
                    value=node_index,
                ),
                # TODO: remove m_val dependency.
                Identifier(description="m_val", position=1, value=m_val),
            ],
        )
        mdsa_snn.add_node(lif_neuron.full_name)
        mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


@typechecked
def create_next_round_node(
    mdsa_snn: nx.DiGraph,
    nr_of_nodes: int,
    run_config: Dict,
    spacing: float,
) -> None:
    """Creates the neuron settings for the counter node in the MDSA
    algorithm."""
    # NOTE, for loop starts at index 1, instead of 0!
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        lif_neuron = LIF_neuron(
            name="next_round",
            bias=0.0,
            du=0.0,
            dv=1.0,
            vth=float(nr_of_nodes) - 1,
            pos=(
                float(6 * spacing + (m_val - 1) * 9 * spacing),
                -2 * spacing,
            ),
            identifiers=[
                Identifier(description="m_val", position=0, value=m_val),
            ],
        )
        mdsa_snn.add_node(lif_neuron.full_name)
        mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]


@typechecked
def create_terminator_node(
    input_graph: nx.Graph, m_val: int, mdsa_snn: nx.DiGraph, spacing: float
) -> None:
    """T800 node that stops the spiking network from proceeding, once the
    algorithm is completed."""
    lif_neuron = LIF_neuron(
        name="terminator_node",
        bias=0.0,
        du=0.0,
        dv=1.0,
        vth=float(len(input_graph.nodes)) - 1,
        pos=(
            float(6 * spacing + (m_val) * 9 * spacing),
            -2 * spacing,
        ),
    )
    mdsa_snn.add_node(lif_neuron.full_name)
    mdsa_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]
