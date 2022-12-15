"""Specifies the plotting layout of the snn."""

from typing import List, Optional

from snnbackends.networkx.LIF_neuron import Identifier
from typeguard import typechecked


class MDSA_layout:
    """Stores plotting layout for neurons in MDSA algorithm."""

    # pylint: disable=R0903
    @typechecked
    def __init__(self) -> None:
        """Checks if the graph is plotted with or without adaptation.

        Based on that information, it specifies the layout of the nodes.
        """
        self.spacing = 0.25


# pylint: disable=R0902
class MDSA_circuit_dimensions:
    """Contains the dimensions of the MDSA circuit for n=1,m=1."""

    @typechecked
    def __init__(self, graph_size: float, redundancy: float) -> None:
        """Gets the layout of the first MDSA circuit with in it the:

        - spike_once
        - rand
        - degree_receiver
        - selector
        - next round
        neurons.
        """
        self.graph_size = graph_size
        self.redundancy = redundancy
        # x0 is the starting x-coordinate of the circuit.
        # y0 is the starting y-coordinate of the circuit.
        self.x0 = 0
        self.y0 = 0
        self.dx_spike_once_degree_receiver = 0.1
        self.dy_spike_once_rand = 0.1
        self.dy_degree_receivers = 0.1

        self.max_height = self.max_circuit_height()
        self.max_width = self.max_circuit_width()

    @typechecked
    def max_circuit_height(
        self,
    ) -> float:
        """Returns the maximum height of the circuit for n=1."""
        return max(
            self.max_circuit_height_rand(), self.max_height_degree_receiver()
        )

    def max_circuit_width(
        self,
    ) -> float:
        """Returns the maximum width of the circuit for m=1."""
        return self.max_width_degree_receiver()

    @typechecked
    def max_circuit_height_spike_once(self) -> float:
        """Returns the max y-coordinate of the spike_once nodes, including
        redundant nodes, for the first circuit.

        # TODO: duplicate code with:max_height_degree_receiver
        """
        spike_once_node = Node_layout("rand")
        return spike_once_node.max_height_redundancy(self.y0, self.redundancy)

    @typechecked
    def max_circuit_width_spike_once(self) -> float:
        """Returns the max x-coordinate of the spike_once nodes, including
        redundant nodes, for the first circuit.

        # TODO: duplicate code with:max_width_degree_receiver
        """
        spike_once_node = Node_layout("rand")
        return spike_once_node.max_width_redundancy(self.x0, self.redundancy)

    @typechecked
    def max_circuit_height_rand(self) -> float:
        """Returns the max y-coordinate of the rand nodes, including redundant
        nodes, for the first circuit."""
        y_0_rand = self.max_circuit_height_spike_once()
        rand_node = Node_layout("rand")
        return self.dy_spike_once_rand + rand_node.max_height_redundancy(
            y_0_rand, self.redundancy
        )

    @typechecked
    def max_height_degree_receiver(self) -> float:
        """Returns the max y-coordinate of the degree_receiver nodes, including
        redundant nodes, for the first circuit."""
        nr_of_degree_receivers: float = self.graph_size - 1
        degree_receiver_node = Node_layout("degree_receiver")
        return nr_of_degree_receivers * (
            degree_receiver_node.max_height_redundancy(0, self.redundancy)
            + self.dy_degree_receivers
        )

    @typechecked
    def max_width_degree_receiver(self) -> float:
        """Returns the max x-coordinate of the degree_receiver nodes, including
        redundant nodes, for the first circuit."""
        spike_once_node = Node_layout("spike_once")
        degree_receiver_node = Node_layout("degree_receiver")
        return (
            spike_once_node.max_width_redundancy(0, self.redundancy)
            + self.dx_spike_once_degree_receiver
            + degree_receiver_node.max_width_redundancy(0, self.redundancy)
        )


class Node_layout:
    """Stores plotting layout for nodes representing neurons in MDSA
    algorithm."""

    # pylint: disable=R0903

    @typechecked
    def __init__(
        self,
        nodename: str,
    ) -> None:
        """A node is assumed to have:

        - a radius.
        - a name written over it horizontally,
        - a rectangle with neuron properties on the top right.
        The props rectangle is assumed to have zero overlap with the radius.
        """
        self.radius: float = 160
        self.name_fontsize: float = 6
        self.props_fontsize: float = 6
        self.name = nodename
        self.name_width = len(self.name) * self.name_fontsize
        self.props_width = 10 * self.props_fontsize
        self.props_height = 20 * self.props_fontsize
        self.eff_width = max(self.radius + self.props_width, self.name_width)
        self.eff_height = self.radius + self.props_width

    @typechecked
    def max_width_redundancy(self, x0: float, redundancy: float) -> float:
        """Returns the maximum width of a redundant node."""
        node = Node_layout(self.name)
        return x0 + node.eff_width * redundancy

    @typechecked
    def max_height_redundancy(self, y0: float, redundancy: float) -> float:
        """Returns the maximum height of a redundant node."""
        node = Node_layout(self.name)
        return y0 + node.eff_height * redundancy


# pylint: disable=R0913
@typechecked
def get_node_position(
    graph_size: float,
    node_name: str,
    identifiers: List[Identifier],
    node_redundancy: float,
    m_val: Optional[int] = None,
    degree_index: Optional[int] = None,
) -> List[float]:
    """Returns the node position."""
    # 0 redundancy is default, 1 redundancy is 1 backup neuron etc.
    # redundancy:int = run_config["adaptation"]["redundancy"] # TODO: apply
    redundancy: float = 1
    circuit = MDSA_circuit_dimensions(graph_size, redundancy)

    if node_name == "spike_once":
        return spike_once_xy(
            circuit=circuit,
            node_index=identifiers[0].value,
            node_redundancy=node_redundancy,
        )

    if node_name == "rand":
        return rand_xy(
            circuit=circuit,
            node_index=identifiers[0].value,
            node_redundancy=node_redundancy,
        )

    if node_name == "degree_receiver":
        return degree_receiver_xy(
            circuit=circuit,
            node_index=identifiers[0].value,
            degree_index_per_circuit=degree_index,
            m_val=m_val,
            node_redundancy=node_redundancy,
        )
    raise Exception(f"Error, node:{node_name} not supported.")


@typechecked
def spike_once_xy(
    circuit: MDSA_circuit_dimensions, node_index: int, node_redundancy: float
) -> List[float]:
    """Returns the bottom left x and y coordinates of a spike_once node."""
    node = Node_layout("spike_once")

    return [
        node.eff_width * (node_redundancy),
        node.eff_height * (node_redundancy) + circuit.max_height * node_index,
    ]


@typechecked
def rand_xy(
    circuit: MDSA_circuit_dimensions, node_index: int, node_redundancy: float
) -> List[float]:
    """Returns the bottom left x and y coordinates of a rand node."""
    node = Node_layout("rand")

    start_height_in_circuit = (
        circuit.max_circuit_height_spike_once() + circuit.dy_spike_once_rand
    )
    return [
        node.eff_width * (node_redundancy),
        start_height_in_circuit
        + node.eff_height * (node_redundancy)
        + circuit.max_height * node_index,
    ]


@typechecked
def degree_receiver_xy(
    circuit: MDSA_circuit_dimensions,
    node_index: int,
    degree_index_per_circuit: int,
    m_val: int,
    node_redundancy: float,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a degree_receiver node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """
    if m_val is None or degree_index_per_circuit is None:
        raise Exception("Error, m_val and degree_index_per_circuit required.")
    node = Node_layout("degree_receiver")
    start_width_in_circuit = (
        circuit.max_circuit_width_spike_once()
        + circuit.dx_spike_once_degree_receiver
    )
    print(f"degree_index_per_circuit={degree_index_per_circuit}")
    print(f"node_redundancy={node_redundancy}")
    print(f"node_index={node_index}")
    # TODO: include m in x coordinate.
    return [
        start_width_in_circuit
        + node.eff_width * (node_redundancy)
        + circuit.max_width * m_val,
        degree_index_per_circuit
        * (node.eff_height + circuit.dy_degree_receivers)
        * circuit.redundancy
        + node.eff_height * (node_redundancy)
        + circuit.max_height * node_index,
    ]
