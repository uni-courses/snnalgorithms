"""Specifies the plotting layout of the snn.

TODO: instead of creating complicated relative positions, create a grid and
pint the neurons on the grid intersections instead.
"""
from typing import List

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
    def __init__(self, graph_size: float, redundancy: int) -> None:
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
        self.dx_spike_once_degree_receiver = 1
        self.dx_degree_receiver_selector = 0.1
        self.dy_spike_once_rand = 0.1
        self.dy_selector_counter = 0.1
        self.dy_degree_receivers = 0.1
        self.dy_circuit = 30
        self.dx_cicruit = 0

        self.max_height = (
            self.max_circuit_height(self.redundancy) + self.dy_circuit
        )
        self.max_width = self.max_circuit_width() + self.dx_cicruit

    @typechecked
    def max_circuit_height(self, circuit_redundancy: int) -> float:
        """Returns the maximum height of the circuit for n=1."""
        return max(
            self.max_circuit_height_rand(circuit_redundancy),
            self.max_height_degree_receiver(circuit_redundancy),
        )

    def max_circuit_width(
        self,
    ) -> float:
        """Returns the maximum width of the circuit for m=1."""
        return self.max_width_degree_receiver()

    @typechecked
    def max_circuit_height_spike_once(self, circuit_redundancy: int) -> float:
        """Returns the max y-coordinate of the spike_once nodes, including
        redundant nodes, for the first circuit.

        # TODO: duplicate code with:max_height_degree_receiver
        """
        spike_once_node = Node_layout("spike_once")
        return spike_once_node.max_height_redundancy(
            circuit_redundancy,
            "spike_once",
            self.y0,
        )

    @typechecked
    def max_circuit_width_spike_once(self) -> float:
        """Returns the max x-coordinate of the spike_once nodes, including
        redundant nodes, for the first circuit.

        # TODO: duplicate code with:max_width_degree_receiver
        """
        spike_once_node = Node_layout("spike_once")
        return spike_once_node.max_width_redundancy(self.x0, self.redundancy)

    @typechecked
    def max_circuit_height_rand(self, circuit_redundancy: int) -> float:
        """Returns the max y-coordinate of the rand nodes, including redundant
        nodes, for the first circuit."""
        y_0_rand = self.max_circuit_height_spike_once(circuit_redundancy)
        rand_node = Node_layout("rand")
        return self.dy_spike_once_rand + rand_node.max_height_redundancy(
            circuit_redundancy, "rand", y_0_rand
        )

    @typechecked
    def max_height_degree_receiver(self, circuit_redundancy: int) -> float:
        """Returns the max y-coordinate of the degree_receiver nodes, including
        redundant nodes, for the first circuit."""
        max_nr_of_degree_receivers: float = self.graph_size - 1
        degree_receiver_node = Node_layout("degree_receiver")
        return max_nr_of_degree_receivers * (
            degree_receiver_node.max_height_redundancy(
                circuit_redundancy, "degree_receiver", self.y0
            )
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
        node_name: str,
    ) -> None:
        """A node is assumed to have:

        - a radius.
        - a name written over it horizontally,
        - a rectangle with neuron properties on the top right.
        The props rectangle is assumed to have zero overlap with the radius.
        """
        # TODO: parameterize
        # TODO: get sepparate spacing coefficients for the redundancy.
        self.radius: float = 30
        self.name_fontsize: float = 3
        self.props_fontsize: float = 3
        self.name = node_name
        # +4 to allow for the red_ prefix for redundant nodes.
        self.name_width = 0.05 * (len(self.name) + 4) * self.name_fontsize
        self.props_width = 0.1 * self.props_fontsize
        self.props_height = 0.2 * self.props_fontsize

        self.eff_width = max(self.radius + self.props_width, self.name_width)
        self.eff_height = self.radius + self.props_width

    @typechecked
    def max_width_redundancy(self, x0: float, redundancy: float) -> float:
        """Returns the maximum width of a redundant node."""
        node = Node_layout(self.name)
        return x0 + node.eff_width * (redundancy + 1)

    @typechecked
    def max_height_redundancy(
        self,
        circuit_redundancy: float,
        node_name: str,
        y0: float,
    ) -> float:
        """Returns the maximum height of a redundant node."""
        node = Node_layout(node_name)
        return y0 + node.eff_height * circuit_redundancy


@typechecked
def spike_once_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    node_index: int,
    node_redundancy: float,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a spike_once node."""
    node = Node_layout("spike_once")

    return [
        node.eff_width * (node_redundancy),
        node.eff_height * (node_redundancy) + circuit.max_height * node_index,
    ]


@typechecked
def rand_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    node_index: int,
    node_redundancy: float,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a rand node."""
    node = Node_layout("rand")

    start_height_in_circuit = (
        circuit.max_circuit_height_spike_once(circuit.redundancy)
        + circuit.dy_spike_once_rand
    )

    return [
        node.eff_width * (node_redundancy),
        start_height_in_circuit
        + node.eff_height * (node_redundancy)
        + circuit.max_height * node_index,
    ]


@typechecked
def degree_receiver_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    node_index: int,
    degree_index_per_circuit: int,
    m_val: int,
    node_redundancy: int,
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


@typechecked
def selector_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    circuit_redundancy: int,
    node_index: int,
    m_val: int,
    node_redundancy: int,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a degree_receiver node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """
    if m_val is None:
        raise Exception("Error, m_val and degree_index_per_circuit required.")

    degree_receiver_node = Node_layout("degree_receiver")
    degree_receiver_x0 = circuit.max_circuit_width_spike_once()
    start_width_in_circuit = (
        degree_receiver_node.max_width_redundancy(
            degree_receiver_x0, circuit_redundancy
        )
        + circuit.dx_degree_receiver_selector
    )

    selector_node = Node_layout("selector")
    return [
        start_width_in_circuit
        + selector_node.eff_width * (node_redundancy)
        + circuit.max_width * m_val,
        selector_node.eff_height * (node_redundancy)
        + circuit.max_height * node_index,
    ]


@typechecked
def counter_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    circuit_redundancy: int,
    node_index: int,
    m_val: int,
    node_redundancy: int,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a counter node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """
    if m_val is None:
        raise Exception("Error, m_val and degree_index_per_circuit required.")

    # Compute starting width.
    degree_receiver_node = Node_layout("degree_receiver")
    degree_receiver_x0 = circuit.max_circuit_width_spike_once()
    start_width_in_circuit = (
        degree_receiver_node.max_width_redundancy(
            degree_receiver_x0, circuit_redundancy
        )
        + circuit.dx_degree_receiver_selector
    )

    # Compute starting height.
    counter_node = Node_layout("counter")

    redundancy_spacing = get_spacing(node_redundancy=node_redundancy)
    return [
        start_width_in_circuit
        + counter_node.eff_width * (redundancy_spacing)
        + circuit.max_width * m_val,
        # start_height_in_circuit
        +counter_node.eff_height * (node_redundancy)
        + circuit.max_height * node_index,
    ]


@typechecked
def next_round_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    circuit_redundancy: int,
    m_val: int,
    node_redundancy: int,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a degree_receiver node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """
    if m_val is None:
        raise Exception("Error, m_val and degree_index_per_circuit required.")

    degree_receiver_node = Node_layout("degree_receiver")
    degree_receiver_x0 = circuit.max_circuit_width_spike_once()
    start_width_in_circuit = (
        degree_receiver_node.max_width_redundancy(
            degree_receiver_x0, circuit_redundancy
        )
        + circuit.dx_degree_receiver_selector
    )

    start_height_in_circuit = (
        circuit.max_circuit_height_spike_once(circuit.redundancy)
        + circuit.dy_spike_once_rand
    )

    next_round_node = Node_layout("next_round")
    return [
        start_width_in_circuit
        + next_round_node.eff_width * (node_redundancy)
        + circuit.max_width * m_val,
        start_height_in_circuit
        + next_round_node.eff_height * (node_redundancy)
        + circuit.max_height * 1,  # move to second row from bottom.
    ]


@typechecked
def connecting_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    circuit_redundancy: int,
    graph_size: int,
    node_redundancy: int,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a degree_receiver node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """

    degree_receiver_node = Node_layout("degree_receiver")
    degree_receiver_x0 = circuit.max_circuit_width_spike_once()
    start_width_in_circuit = (
        degree_receiver_node.max_width_redundancy(
            degree_receiver_x0, circuit_redundancy
        )
        + circuit.dx_degree_receiver_selector
    )

    start_height_in_circuit = (
        circuit.max_circuit_height_spike_once(circuit.redundancy)
        + circuit.dy_spike_once_rand
    )

    connector_node = Node_layout("connecting")

    redundancy_spacing = get_spacing(node_redundancy=node_redundancy)
    return [
        start_width_in_circuit + connector_node.eff_width * (node_redundancy),
        start_height_in_circuit
        + connector_node.eff_height * (redundancy_spacing)
        + circuit.max_height * (graph_size - 1),
    ]


@typechecked
def terminator_xy(
    *,
    circuit: MDSA_circuit_dimensions,
    circuit_redundancy: int,
    graph_size: int,
    m_val: int,
    node_redundancy: int,
) -> List[float]:
    """Returns the bottom left x and y coordinates of a degree_receiver node.

    The degree_index_per_circuit indicates the position of the
    degree_receiver within the circuit.
    """

    # Compute starting width.
    degree_receiver_node = Node_layout("degree_receiver")
    degree_receiver_x0 = circuit.max_circuit_width_spike_once()
    start_width_in_circuit = (
        degree_receiver_node.max_width_redundancy(
            degree_receiver_x0, circuit_redundancy
        )
        + circuit.dx_degree_receiver_selector
    )

    start_height_in_circuit = (
        circuit.max_circuit_height_spike_once(circuit.redundancy)
        + circuit.dy_spike_once_rand
    )

    terminator_node = Node_layout("terminator")
    redundancy_spacing = get_spacing(node_redundancy=node_redundancy)
    return [
        start_width_in_circuit
        + terminator_node.eff_width * (node_redundancy)
        + circuit.max_width * (m_val - 1),  # Shift 1 m_val to left.
        start_height_in_circuit
        + terminator_node.eff_height * (redundancy_spacing)
        + circuit.max_height * (graph_size - 1),
    ]


def get_hori_redundant_redundancy_spacing(*, bare_node_name: str) -> float:
    """Returns the horizontal spacing that is relevant for redundancy.

    For example, the rand neurons should not be spaced horizontally
    right of the rand neurons, but right of the spike_once neurons.
    """
    if bare_node_name in ["rand", "spike_once"]:
        widest_node_name = "spike_once"
    elif bare_node_name in ["degree_receiver"]:
        widest_node_name = "degree_receiver"
    else:
        widest_node_name = "degree_receiver"
    node_layout = Node_layout(widest_node_name)
    return node_layout.eff_width


def get_spacing(*, node_redundancy: int) -> int:
    """Returns an extra spacing in case there is no redundancy.

    Used to prevent overlapping node positions.
    """
    if node_redundancy == 0:
        redundancy_spacing = 1
    else:
        redundancy_spacing = node_redundancy
    return redundancy_spacing
