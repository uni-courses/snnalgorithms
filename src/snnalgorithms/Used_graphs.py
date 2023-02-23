"""Contains the list of graphs that are used for radiation testing."""
from typing import List

import networkx as nx
from typeguard import typechecked


class Used_graphs:
    """Creates graphs used for paper."""

    @typechecked
    def __init__(self) -> None:
        self.three = self.get_graphs_with_3_neurons()
        self.four = self.get_graphs_with_4_neurons()
        self.five = self.get_graphs_with_5_neurons()

    @typechecked
    def get_graphs(self, size: int) -> List[nx.Graph]:
        """Returns the graphs that are used for testing, per selected size.

        :param size: Nr of nodes in the original graph on which test is ran.
        """
        if size == 3:
            return self.three
        if size == 4:
            return self.four
        if size == 5:
            return self.five
        raise ValueError(f"Graph size of: {size} is currently not supported.")

    @typechecked
    def get_graphs_with_3_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 3."""
        return [self.three_a()]

    @typechecked
    def get_graphs_with_4_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 4."""
        return [self.four_a(), self.four_b(), self.four_c()]

    @typechecked
    def get_graphs_with_5_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 5."""
        return [
            self.five_a(),
            self.five_b(),
            self.five_c(),
            self.five_d(),
            self.five_e(),
            self.five_f(),
        ]

    @typechecked
    def three_a(self) -> nx.Graph:
        """Creates two different graphs of size 3."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
            ]
        )
        return graph

    @typechecked
    def four_a(self) -> nx.Graph:
        """Straight line."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
            ]
        )
        return graph

    @typechecked
    def four_b(self) -> nx.Graph:
        """Y"""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (2, 3),
            ]
        )
        return graph

    @typechecked
    def four_c(self) -> nx.Graph:
        """Square."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ]
        )
        return graph

    @typechecked
    def five_a(self) -> nx.Graph:
        """Straight line."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_b(self) -> nx.Graph:
        """Y-long-tail."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_c(self) -> nx.Graph:
        """Y with 3 arms."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (3, 2),
                (4, 2),
            ]
        )
        return graph

    @typechecked
    def five_d(self) -> nx.Graph:
        """Pentagon."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
            ]
        )
        return graph

    @typechecked
    def five_e(self) -> nx.Graph:
        """Square-with-tail."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_f(self) -> nx.Graph:
        """Square."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ]
        )
        return graph
