"""Specifies the plotting layout of the snn.

TODO: instead of creating complicated relative positions, create a grid and
pint the neurons on the grid intersections instead.
"""
from typing import Dict, List, Optional, Tuple

from snnbackends.networkx.LIF_neuron import Identifier
from snncompare.export_plots.Plot_config import Plot_config
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked


@typechecked
def connecting_xy() -> Tuple[float, float]:
    """Returns the  x and y coordinates of the connector node."""
    return 0.0, 0.0


def get_cumulative_starting_height(
    degree_indices: Dict[int, int],
    dy_node: float,
    node_index: int,
    node_redundancy: int,
    plot_config: Plot_config,
) -> float:
    """Returns the cumulative starting height for a node at some node index
    level."""
    if node_index == 0:
        return 0.0
    sum_height: float = 0
    for i in range(0, node_index):
        sum_height += 1 * dy_node + plot_config.dy_redundant * (
            node_redundancy + (degree_indices[i] - 1)
        )
    return sum_height


@typechecked
def spike_once_xy(
    dx_node: float,
    plot_config: "Plot_config",
    node_redundancy: float,
    sum_height: float,
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a spike_once node."""
    x = dx_node * 1.0 + plot_config.dx_redundant * node_redundancy
    y = sum_height + plot_config.dy_redundant * node_redundancy
    return x, y


@typechecked
def rand_xy(
    dx_node: float,
    plot_config: "Plot_config",
    node_redundancy: float,
    sum_height: float,
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a spike_once node."""
    x = dx_node * 2.0 + plot_config.dx_redundant * node_redundancy
    y = sum_height + plot_config.dy_redundant * node_redundancy
    return x, y


# pylint: disable=R0913
@typechecked
def degree_receiver_xy(
    dx_node: float,
    plot_config: "Plot_config",
    degree_index_per_circuit: int,
    m_val: int,
    node_redundancy: float,
    sum_height: float,
) -> Tuple[float, float]:
    """Returns the x and y coordinates of a degree_receiver node The degree
    index indicates how much higher a ."""
    x = dx_node * (3 + 2 * m_val) + plot_config.dx_redundant * node_redundancy
    y = (
        sum_height
        + degree_index_per_circuit * plot_config.dy_redundant
        + plot_config.dy_redundant * (node_redundancy)
    )
    return x, y


# pylint: disable=R0913
@typechecked
def selector_xy(
    dx_node: float,
    dy_node: float,
    m_val: int,
    node_redundancy: float,
    plot_config: "Plot_config",
    sum_height: float,
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a selector node."""
    x = dx_node * (4 + 2 * m_val) + plot_config.dx_redundant * node_redundancy
    # Move selector nodes up with 1*dy_node to allow next_round node at y=0.
    y = sum_height + 1 * dy_node + plot_config.dy_redundant * node_redundancy
    return x, y


# pylint: disable=R0913
@typechecked
def counter_xy(
    dx_node: float,
    m_val_max: int,
    node_redundancy: float,
    plot_config: "Plot_config",
    sum_height: float,
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a counter node."""
    x = (
        dx_node * (5 + 2 * m_val_max)
        + plot_config.dx_redundant * node_redundancy
    )
    # y = dy_node * node_index + plot_config.dy_redundant * node_redundancy
    y = sum_height + plot_config.dy_redundant * node_redundancy
    return x, y


@typechecked
def terminator_xy(
    dx_node: float,
    m_val_max: int,
    node_redundancy: float,
    plot_config: "Plot_config",
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a terminator node."""
    x = (
        dx_node * (6 + 2 * m_val_max)
        + plot_config.dx_redundant * node_redundancy
    )
    y = 0
    return x, y


# pylint: disable=R0913
@typechecked
def next_round_xy(
    dx_node: float,
    m_val: int,
    node_redundancy: float,
    plot_config: "Plot_config",
) -> Tuple[float, float]:
    """Returns the  x and y coordinates of a next_round node.

    Same x-coordinate as the selector node, except at y=0
    """
    x = dx_node * (4 + 2 * m_val) + plot_config.dx_redundant * node_redundancy
    y = 0 + plot_config.dy_redundant * node_redundancy
    return (x, y)


# pylint: disable=R0911
# pylint: disable=R0913
@typechecked
def get_node_position(
    *,
    node_name: str,
    identifiers: List[Identifier],
    node_redundancy: int,
    plot_config: Plot_config,
    run_config: Run_config,
    m_val_max: Optional[int] = None,
    degree_index: Optional[int] = None,
    degree_indices: Optional[Dict[int, int]] = None,
) -> Tuple[float, float]:
    """Returns the node position."""
    # 0 redundancy is default, 1 redundancy is 1 backup neuron etc.
    redundancy: int
    if run_config.adaptation is None:
        redundancy = 0
    else:
        redundancy = run_config.adaptation.redundancy

    if degree_indices is not None:
        if identifiers[0].description != "node_index":
            raise ValueError("Error, node_index not found.")
        sum_height: float = (
            get_cumulative_starting_height(  # type:ignore[arg-type]
                degree_indices=degree_indices,
                dy_node=plot_config.y_node_spacer,
                node_index=identifiers[0].value,
                node_redundancy=redundancy,
                plot_config=plot_config,
            )
        )

    dx_node = redundancy * plot_config.dx_redundant + plot_config.x_node_spacer
    dy_node = redundancy * plot_config.dx_redundant + plot_config.x_node_spacer

    # At most n-1 degree_receivers per node (circuit).
    # Put them above each other in a circuit.
    # dy_node = (
    # graph_size - 1 + redundancy
    # ) * plot_config.dy_redundant + plot_config.y_node_spacer

    if node_name == "spike_once":
        return spike_once_xy(
            dx_node=dx_node,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
            sum_height=sum_height,
        )

    if node_name == "rand":
        return rand_xy(
            dx_node=dx_node,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
            sum_height=sum_height,
        )

    if node_name == "degree_receiver":
        return degree_receiver_xy(
            dx_node=dx_node,
            degree_index_per_circuit=degree_index,
            m_val=identifiers[2].value,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
            sum_height=sum_height,
        )
    if node_name == "selector":
        return selector_xy(
            dx_node=dx_node,
            dy_node=dy_node,
            m_val=identifiers[1].value,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
            sum_height=sum_height,
        )

    if node_name == "counter":
        return counter_xy(
            dx_node=dx_node,
            m_val_max=m_val_max,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
            sum_height=sum_height,
        )

    if node_name == "next_round":
        return next_round_xy(
            dx_node=dx_node,
            m_val=identifiers[0].value,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
        )

    if node_name == "connecting":
        return connecting_xy()

    if node_name == "terminator":
        return terminator_xy(
            dx_node=dx_node,
            m_val_max=m_val_max,
            node_redundancy=node_redundancy,
            plot_config=plot_config,
        )

    raise ValueError(f"Error, node:{node_name} not supported.")
