"""Creates the MDSA snn synapses."""


from typing import Dict

import networkx as nx
from snnbackends.networkx.LIF_neuron import Synapse
from typeguard import typechecked


@typechecked
def create_MDSA_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
) -> nx.DiGraph:
    """Creates the synapses between the neurons for the MDSA algorithm."""

    # Create synapses for connecting node.
    create_outgoing_connecting_synapses(input_graph, mdsa_snn)

    # Create spike_once nodes.
    for _ in input_graph.nodes:
        create_outgoing_spike_once_synapses(input_graph, mdsa_snn)

    create_degree_receiver_selector_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    # pylint: disable=R0801
    create_degree_receiver_counter_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    create_degree_receiver_next_round_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    create_outgoing_selector_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    create_outgoing_rand_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    create_degree_to_degree_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    create_outgoing_next_round_synapses(
        mdsa_snn,
        run_config,
    )
    create_outgoing_d_charger_synapses(
        mdsa_snn,
        run_config,
    )
    create_outgoing_delay_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )

    # pylint: disable=R0801
    create_degree_receiver_terminator_synapses(
        input_graph,
        mdsa_snn,
        run_config,
    )
    return mdsa_snn


@typechecked
def create_outgoing_connecting_synapses(
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
                    "connecting_node",
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
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
) -> None:
    """Creates the outgoing synapses for the spike_once node in the MDSA
    algorithm."""
    rand_ceil = input_graph.graph["alg_props"]["rand_ceil"]
    delta = input_graph.graph["alg_props"]["delta"]
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
                                weight=rand_ceil * delta,
                                delay=0,
                                change_per_t=0,
                            ),
                        )


@typechecked
def create_degree_receiver_selector_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create inhibitory synapse to selector.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                for m_val in range(
                    0, run_config["algorithm"]["MDSA"]["m_val"] + 1
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
                            weight=-5,
                            delay=0,
                            change_per_t=0,
                        ),  # to disable bias
                    )


def create_degree_receiver_counter_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create synapse to counter neuron.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                # TODO: Remove the m_val dependency
                m_subscript = max(0, run_config["algorithm"]["MDSA"]["m_val"])
                mdsa_snn.add_edges_from(
                    [
                        (
                            f"degree_receiver_{node_index}_"
                            # TODO: eliminate dependency on m_val
                            + f"{neighbour_index}_{m_subscript}",
                            f"counter_{neighbour_index}_{m_subscript}",
                        )
                    ],
                    synapse=Synapse(
                        weight=1,
                        delay=0,
                        change_per_t=0,
                    ),  # Used to disable bias.
                )


def create_degree_receiver_next_round_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
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
                        1, run_config["algorithm"]["MDSA"]["m_val"] + 1
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
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
) -> None:
    """Creates the outgoing synapses for the rand node in the MDSA
    algorithm."""
    # Add synapse from selector node back into degree selector.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                for m_val in range(
                    0, run_config["algorithm"]["MDSA"]["m_val"] + 1
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
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
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
                        0, run_config["algorithm"]["MDSA"]["m_val"] + 1
                    ):
                        mdsa_snn.add_edges_from(
                            [
                                (
                                    f"rand_{node_index}_{m_val}",
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{node_index}_{m_val}",
                                )
                            ],
                            synapse=Synapse(
                                weight=input_graph.graph["alg_props"][
                                    "initial_rand_current"
                                ][node_index],
                                delay=0,
                                change_per_t=0,
                            ),
                        )


@typechecked
def create_degree_to_degree_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
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
    for m_val in range(0, run_config["algorithm"]["MDSA"]["m_val"] + 1):
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
def create_outgoing_next_round_synapses(
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
) -> None:
    """Creates the outgoing synapses for the next_round node in the MDSA
    algorithm."""

    # Create outgoing synapses
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        mdsa_snn.add_edges_from(
            [
                (
                    f"next_round_{m_val}",
                    f"d_charger_{m_val}",
                )
            ],
            synapse=Synapse(
                weight=1,
                delay=0,
                change_per_t=0,
            ),
        )


@typechecked
def create_outgoing_d_charger_synapses(
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
) -> None:
    """Creates the outgoing synapses for the d_charger node in the MDSA
    algorithm."""

    # Create outgoing synapses
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        mdsa_snn.add_edges_from(
            [
                (
                    f"d_charger_{m_val}",
                    f"delay_{m_val}",
                )
            ],
            synapse=Synapse(
                weight=+1,
                delay=0,
                change_per_t=0,
            ),
        )


@typechecked
def create_outgoing_delay_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: Dict,
) -> None:
    """Creates the outgoing synapses for the d_charger node in the MDSA
    algorithm.

    TODO: merge with:
    create_outgoing_next_round_synapses
    create_outgoing_d_charger_synapses
    """

    # Create outgoing synapses
    for m_val in range(1, run_config["algorithm"]["MDSA"]["m_val"] + 1):
        mdsa_snn.add_edges_from(
            [
                (
                    f"delay_{m_val}",
                    f"d_charger_{m_val}",
                )
            ],
            synapse=Synapse(
                weight=-100,
                delay=0,
                change_per_t=0,
            ),
        )

        for node_index in input_graph.nodes:
            mdsa_snn.add_edges_from(
                [
                    (
                        f"delay_{m_val}",
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
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    run_config: dict,
) -> None:
    """Creates the outgoing synapses for the degree_receiver node in the MDSA
    algorithm."""

    # Create synapse to terminator neuron.
    for node_index in input_graph.nodes:
        for neighbour_index in nx.all_neighbors(input_graph, node_index):
            if node_index != neighbour_index:
                # TODO: Remove the m_val dependency
                m_subscript = max(0, run_config["algorithm"]["MDSA"]["m_val"])
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
