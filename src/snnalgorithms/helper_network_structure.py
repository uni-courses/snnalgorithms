"""Assists the conversion from the input graph to an SNN graph that performs
the MDSA approximation."""
from typing import List

import networkx as nx
from snncompare.export_plots.get_plot_data import add_recursive_edges_to_graph
from typeguard import typechecked


@typechecked
def create_synapses_and_spike_dicts(
    input_graph: nx.Graph,
    get_degree: nx.DiGraph,
    left: List[dict],
    m: int,
    rand_ceil: float,
    right: List[dict],
) -> None:
    """Creates some synapses and the spike dictionary."""
    # pylint: disable=R0913
    # 6/5 arguments are currently used in the synapse creation method.
    # TODO: add recurrent synapses (as edges).
    add_recursive_edges_to_graph(get_degree)

    # Create replacement synapses.
    if m <= 1:
        # TODO: remove return get_degree
        get_degree = create_degree_synapses_for_m_is_zero(
            get_degree, left, m, rand_ceil, right
        )
    else:
        get_degree = retry_create_degree_synapses(
            input_graph, get_degree, m, rand_ceil
        )

    # Create spike dictionaries with [t] as key, and boolean spike as value for
    # each node.
    for node in get_degree.nodes:
        get_degree.nodes[node]["spike"] = {}


@typechecked
def create_degree_synapses_for_m_is_zero(
    get_degree: nx.DiGraph,
    left: List[dict],
    m: int,
    rand_ceil: float,
    right: List[dict],
) -> nx.DiGraph:
    """

    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param left:
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param rand_ceil: Ceiling of the range in which rand nrs can be generated.
    :param right:

    """
    # pylint: disable=R1702
    # Currently no method is found to reduce the 6/5 nested blocks.
    for some_id in range(m - 1):
        for l_key, l_value in left[some_id].items():
            for l_counter in l_value:
                for r_key, r_value in right[some_id].items():
                    for r_degree in r_value:
                        if l_counter == r_key:
                            get_degree.add_edges_from(
                                [
                                    (
                                        l_key,
                                        r_degree,
                                    )
                                ],
                                weight=rand_ceil,  # Increase u(t) at each t.
                            )
    return get_degree


@typechecked
def retry_create_degree_synapses(
    input_graph: nx.Graph, get_degree: nx.DiGraph, m: int, rand_ceil: float
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
    for loop in range(0, m):
        for x_l in input_graph.nodes:
            for y in input_graph.nodes:
                for x_r in input_graph.nodes:
                    if (
                        f"degree_receiver_{x_l}_{y}_{loop}" in get_degree.nodes
                        and (
                            f"degree_receiver_{x_r}_{y}_{loop+1}"
                            in get_degree.nodes
                        )
                    ):
                        get_degree.add_edges_from(
                            [
                                (
                                    f"degree_receiver_{x_l}_{y}_{loop}",
                                    f"degree_receiver_{x_r}_{y}_{loop+1}",
                                )
                            ],
                            weight=rand_ceil,  # Increase u(t) at each t.
                        )
    return get_degree
