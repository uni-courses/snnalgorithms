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
        self.dx_spike_once_degree_receiver = 1
        self.dy_spike_once_rand = 0.1
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
        nodename: str,
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
        self.name = nodename
        # +4 to allow for the red_ prefix for redundant nodes.
        self.name_width = 0.05 * (len(self.name) + 4) * self.name_fontsize
        self.props_width = 0.1 * self.props_fontsize
        self.props_height = 0.2 * self.props_fontsize
        # print(f'self.name={self.name}')
        # print(f'radius + self.props_width={self.radius + self.props_width}')
        # print(f'self.name_width={self.name_width}')
        # exit()
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
        nodename: str,
        y0: float,
    ) -> float:
        """Returns the maximum height of a redundant node."""
        node = Node_layout(nodename)
        return y0 + node.eff_height * circuit_redundancy


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
        circuit.max_circuit_height_spike_once(circuit.redundancy)
        + circuit.dy_spike_once_rand
    )
    print(f"start_height_in_circuit={start_height_in_circuit}")
    print(f"circuit.max_height={circuit.max_height}")
    print(f"node_index={node_index}")
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


def get_hori_redundant_redundancy_spacing(bare_nodename: str) -> float:
    """Returns the horizontal spacing that is relevant for redundancy.

    For example, the rand neurons should not be spaced horizontally
    right of the rand neurons, but right of the spike_once neurons.
    """
    if bare_nodename in ["rand", "spike_once"]:
        widest_nodename = "spike_once"
    elif bare_nodename in ["degree_receiver"]:
        widest_nodename = "degree_receiver"
    else:
        widest_nodename = "degree_receiver"
    node_layout = Node_layout(widest_nodename)
    return node_layout.eff_width
