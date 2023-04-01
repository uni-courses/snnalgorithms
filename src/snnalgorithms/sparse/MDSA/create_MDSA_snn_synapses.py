"""Creates the MDSA snn synapses."""

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron, Synapse
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked


@typechecked
def create_MDSA_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> nx.DiGraph:
    """Creates the synapses between the neurons for the MDSA algorithm."""

    # Create synapses for connecting node.
    create_outgoing_connecting_synapses(
        input_graph=input_graph, mdsa_snn=mdsa_snn
    )

    # Create spike_once nodes.
    for _ in input_graph.nodes:
        create_outgoing_spike_once_synapses(
            input_graph=input_graph, mdsa_snn=mdsa_snn
        )

    create_degree_receiver_selector_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    # pylint: disable=R0801
    create_degree_receiver_counter_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    create_degree_receiver_next_round_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    create_outgoing_selector_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    create_outgoing_rand_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    create_degree_to_degree_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )

    create_outgoing_next_round_selector_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )
    # pylint: disable=R0801
    create_degree_receiver_terminator_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )
    create_degree_receiver_inhibitory_synapses(
        input_graph=input_graph,
        mdsa_snn=mdsa_snn,
        run_config=run_config,
    )
    return mdsa_snn


@typechecked
def create_outgoing_connecting_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""

    # Create outgoing synapses
    for node_index in input_graph.nodes:
        mdsa_snn.add_edges_from(
            [
                (
                    "connector_node",
                    f"spike_once_{node_index}",
                )
            ],
            synapse=Synapse(
                weight=0,
                delay=0,
                change_per_t=0,
            ),
        )


@typechecked
def create_outgoing_spike_once_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
) -> None:
    """Creates the outgoing synapses for the spike_once node in the MDSA
    algorithm."""
    rand_ceil = input_graph.graph["alg_props"]["rand_ceil"]
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                for other_node_index in input_graph.nodes:
                    if input_graph.has_edge(neighbour_index, other_node_index):
                        mdsa_snn.add_edges_from(
                            [
                                (
                                    f"spike_once_{other_node_index}",
                                    f"degree_receiver_{node_index}_"
                                    + f"{neighbour_index}_0",
                                )
                            ],
                            synapse=Synapse(
                                weight=rand_ceil,
                                delay=0,
                                change_per_t=0,
                            ),
                        )


@typechecked
def create_degree_receiver_selector_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create inhibitory synapse to selector.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                for m_val in range(
                    0, run_config.algorithm["MDSA"]["m_val"] + 1
                ):
                    mdsa_snn.add_edges_from(
                        [
                            (
                                f"degree_receiver_{node_index}_"
                                + f"{neighbour_index}_{m_val}",
                                f"selector_{node_index}_{m_val}",
                            )
                        ],
                        synapse=Synapse(
                            weight=-100,
                            delay=0,
                            change_per_t=0,
                        ),  # to disable bias
                    )


def create_degree_receiver_counter_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create synapse to counter neuron.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                # TODO: Remove the m_val dependency
                m_subscript = max(0, run_config.algorithm["MDSA"]["m_val"])
                mdsa_snn.add_edges_from(
                    [
                        (
                            f"degree_receiver_{node_index}_"
                            + f"{neighbour_index}_{m_subscript}",
                            f"counter_{neighbour_index}",
                        )
                    ],
                    synapse=Synapse(
                        weight=1,
                        delay=0,
                        change_per_t=0,
                    ),  # Used to disable bias.
                )


def create_degree_receiver_next_round_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create synapse to next_round node.
    for node_index in input_graph.nodes:
        for circuit_target in input_graph.nodes:
            if node_index != circuit_target:
                # Check if there is an edge from neighbour_a to neighbour_b.
                if node_index in nx.all_neighbors(input_graph, circuit_target):
                    for m_val in range(
                        1, run_config.algorithm["MDSA"]["m_val"] + 1
                    ):
                        mdsa_snn.add_edges_from(
                            [
                                (
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{node_index}_{m_val-1}",
                                    f"next_round_{m_val}",
                                )
                            ],
                            synapse=Synapse(
                                weight=1,
                                delay=0,
                                change_per_t=0,
                            ),
                        )


@typechecked
def create_outgoing_selector_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the rand node in the MDSA
    algorithm."""
    # Add synapse from selector node back into degree selector.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                for m_val in range(
                    0, run_config.algorithm["MDSA"]["m_val"] + 1
                ):
                    mdsa_snn.add_edges_from(
                        [
                            (
                                f"selector_{node_index}_{m_val}",
                                f"degree_receiver_{node_index}_"
                                + f"{neighbour_index}_{m_val}",
                            )
                        ],
                        synapse=Synapse(
                            weight=1,
                            delay=0,
                            change_per_t=0,
                        ),
                    )


@typechecked
def create_outgoing_rand_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the selector node in the MDSA
    algorithm."""
    # Add synapse between selectorom node and degree receiver nodes.
    for node_index in input_graph.nodes:
        for circuit_target in input_graph.nodes:
            if node_index != circuit_target:
                # Check if there is an edge from neighbour_a to neighbour_b.
                if node_index in nx.all_neighbors(input_graph, circuit_target):
                    for m_val in range(
                        0, run_config.algorithm["MDSA"]["m_val"] + 1
                    ):
                        mdsa_snn.add_edges_from(
                            [
                                (
                                    f"rand_{node_index}",
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{node_index}_{m_val}",
                                )
                            ],
                            synapse=Synapse(
                                weight=input_graph.graph["alg_props"][
                                    "rand_edge_weights"
                                ][node_index],
                                delay=0,
                                change_per_t=0,
                            ),
                        )


@typechecked
def create_degree_to_degree_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> nx.DiGraph:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param rand_ceil: Ceiling of the range in which rand nrs can be generated.

    """
    # pylint: disable=R0913
    # Currently no method is found to reduce the 6/5 nested blocks.
    rand_ceil = input_graph.graph["alg_props"]["rand_ceil"]
    for m_val in range(0, run_config.algorithm["MDSA"]["m_val"] + 1):
        for node_index_left in input_graph.nodes:
            for y in input_graph.nodes:
                for node_index_right in input_graph.nodes:
                    if (
                        f"degree_receiver_{node_index_left}_{y}_{m_val}"
                        in mdsa_snn.nodes
                        and (
                            f"degree_receiver_{node_index_right}_{y}_{m_val+1}"
                            in mdsa_snn.nodes
                        )
                    ):
                        mdsa_snn.add_edges_from(
                            [
                                (
                                    f"degree_receiver_{node_index_left}_"
                                    + f"{y}_{m_val}",
                                    f"degree_receiver_{node_index_right}_"
                                    + f"{y}_{m_val+1}",
                                )
                            ],
                            synapse=Synapse(
                                weight=rand_ceil,  # Increase u(t) at each t.
                                delay=0,
                                change_per_t=0,
                            ),
                        )
    return mdsa_snn


@typechecked
def create_outgoing_next_round_selector_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the d_charger node in the MDSA
    algorithm.

    TODO: merge with:
    create_outgoing_next_round_synapses
    create_outgoing_d_charger_synapses
    """

    # Create outgoing synapses
    for m_val in range(1, run_config.algorithm["MDSA"]["m_val"] + 1):
        for node_index in input_graph.nodes:
            mdsa_snn.add_edges_from(
                [
                    (
                        f"next_round_{m_val}",
                        f"selector_{node_index}_{m_val}",
                    )
                ],
                synapse=Synapse(
                    weight=1,
                    delay=0,
                    change_per_t=0,
                ),
            )


def create_degree_receiver_terminator_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create synapse to terminator neuron.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                # TODO: Remove the m_val dependency
                m_subscript = max(0, run_config.algorithm["MDSA"]["m_val"])
                mdsa_snn.add_edges_from(
                    [
                        (
                            f"degree_receiver_{node_index}_"
                            # TODO: eliminate dependency on m_val
                            + f"{neighbour_index}_{m_subscript}",
                            "terminator_node",
                        )
                    ],
                    synapse=Synapse(
                        weight=1,
                        delay=0,
                        change_per_t=0,
                    ),  # Used to disable bias.
                )


def create_degree_receiver_inhibitory_synapses(
    *,
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm.

    These inhibitory synapses are used to directly silence any other
    competing degree_receiver neurons in that circuit, once a winner has
    been found.
    """

    # Create synapse to inhibitory neuron.
    # pylint: disable=R1702
    for node_index in input_graph.nodes:
        for m_val in range(0, run_config.algorithm["MDSA"]["m_val"] + 1):
            circuit_degree_receivers = []
            for node_name in mdsa_snn.nodes:
                deg_lif = mdsa_snn.nodes[node_name]["nx_lif"][0]

                if deg_lif.name == "degree_receiver":
                    # Get the degree_receivers with the correct index.
                    if (
                        get_identifier_value(lif_neuron=deg_lif, position=0)
                        == node_index
                    ):
                        if (
                            get_identifier_value(
                                lif_neuron=deg_lif, position=2
                            )
                            == m_val
                        ):
                            circuit_degree_receivers.append(deg_lif)
            # Within all the degree receivers of a single circuit, set create
            # the inhibitory synapses.
            for deg_lif in circuit_degree_receivers:
                for other_deg_lif in circuit_degree_receivers:
                    if deg_lif != other_deg_lif:
                        mdsa_snn.add_edges_from(
                            [(deg_lif.full_name, other_deg_lif.full_name)],
                            synapse=Synapse(
                                weight=-100,
                                delay=0,
                                change_per_t=0,
                            ),
                        )


def get_identifier_value(*, lif_neuron: LIF_neuron, position: int) -> int:
    """Returns the identifier value of a Lif neuron at the desired position.

    The positions represent the subscript indice positions in the neuron
    names. For example: degree_receiver_5_6_4 has at identifier position
    0 a value of 5, at identifier position 2 a value of 4.
    """

    for identifier in lif_neuron.identifiers:
        if identifier.position == position:
            return identifier.value
    raise AttributeError(
        f"Identifier position:{position} not found in node:"
        + f"{lif_neuron.full_name}."
    )
