"""Computes which nodes are selected by the MDSA algorithm presented by Alipour
et al."""
from typing import Dict

import networkx as nx
from snncompare.helper import (
    compute_mark,
    compute_marks_for_m_larger_than_one,
    set_node_default_values,
)
from typeguard import typechecked


# pylint: disable=R0913
@typechecked
def get_results(
    *,
    input_graph: nx.Graph,
    m_val: int,
    rand_props: Dict,
    seed: int,
    size: int,
) -> Dict[str, int]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param rand_props:

    """

    rand_ceil = rand_props["rand_ceil"]
    # Reverse list cause the random numbers are subtracted in the edge weights.
    # That causes the highest to "fire first" and lowest to fire last, so the
    # mark order is swapped. Hence the [::-1]
    rand_nrs = rand_props["rand_nrs"][::-1]

    for node in input_graph.nodes:
        set_node_default_values(
            input_graph=input_graph,
            node=node,
            rand_ceil=rand_ceil,
            uninhibited_spread_rand_nrs=rand_nrs,
        )

    # pylint: disable=R0801
    compute_mark(input_graph=input_graph, rand_ceil=rand_ceil)

    compute_marks_for_m_larger_than_one(
        input_graph=input_graph,
        m=m_val,
        seed=seed,
        size=size,
        rand_ceil=rand_ceil,
        export=False,
        show=False,
    )
    counter_marks = {}
    for node_index in input_graph.nodes:
        counter_marks[f"counter_{node_index}"] = input_graph.nodes[node_index][
            "countermarks"
        ]
    return counter_marks
