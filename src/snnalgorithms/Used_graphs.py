"""Contains the list of graphs that are used for radiation testing."""
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_input_graph_output_filepath,
)
from snncompare.import_results.helper import get_isomorphic_graph_hash
from typeguard import typechecked


class Used_graphs:
    """Creates graphs used for paper."""

    @typechecked
    def __init__(self) -> None:
        self.three = self.get_graphs_with_3_neurons()
        self.four = self.get_graphs_with_4_neurons()
        self.five = self.get_graphs_with_5_neurons()

    @typechecked
    def get_graphs(
        self, max_nr_of_graphs: int, seed: int, size: int
    ) -> Dict[str, nx.Graph]:
        """Returns the graphs that are used for testing, per selected size.

        :param size: Nr of nodes in the original graph on which test is ran.
        """
        # if size == 3:
        # return self.three
        # if size == 4:
        # return self.four
        # if size == 5:
        # return self.five
        return get_rand_planar_triangle_free_graph(
            density_cutoff=0.01,
            max_nr_of_graphs=max_nr_of_graphs,
            seed=seed,
            size=size,
        )

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


def triangle_free_graph(size: int, seed: int) -> nx.Graph:
    """Construct a triangle free graph."""
    nodes = range(size)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    edge_candidates = list(combinations(nodes, 2))
    random.seed(seed)
    random.shuffle(edge_candidates)
    for u, v in edge_candidates:
        # Add all triangle free edges.
        if not set(g.neighbors(u)) & set(g.neighbors(v)):
            g.add_edge(u, v)
            if sum(nx.triangles(g).values()) > 0:
                raise ValueError("Got triangular graph.")

            # If adding the edge removes planarity property, remove the edge.
            if not nx.is_planar(g):
                g.remove_edge(u, v)
    return g


def get_rand_planar_triangle_free_graph(
    density_cutoff: float,
    max_nr_of_graphs: int,
    seed: int,
    size: int,
    max_iterations: Optional[int] = 10000,
) -> Dict[str, nx.Graph]:
    """Generates unique, random, undirected, connected, planar, triangle-free
    graphs, and returns them in a list."""
    input_graphs: Dict[str, nx.Graph] = {}
    input_graph: nx.Graph = triangle_free_graph(seed=seed, size=size)

    # Limit the maximum number of times a new graph is tried/created.
    for iteration in range(max_iterations):  # type:ignore[arg-type]
        # If enough graphs are found, do not continue.
        if len(input_graphs) < max_nr_of_graphs:
            # Get a new planar, triangle free graph.
            # iteration+seed because a new graph needs to be tried each call.
            input_graph = triangle_free_graph(seed=iteration + seed, size=size)
            # Verify the density, connectedness and planarity.
            if (
                nx.density(input_graph) > density_cutoff
                and nx.is_connected(input_graph)
                and nx.is_planar(input_graph)
                and sum(nx.triangles(input_graph).values()) == 0
            ):
                isomorphic_hash: str = get_isomorphic_graph_hash(
                    some_graph=input_graph
                )

                # If input graph exists, load it from file.
                output_filepath: str = get_input_graph_output_filepath(
                    input_graph=input_graph
                )
                if Path(output_filepath).is_file():
                    # Load graph from file and verify it results in the same
                    # graph.
                    with open(output_filepath, encoding="utf-8") as json_file:
                        some_json_graph = json.load(json_file)
                        json_file.close()
                        if (
                            "completed_stages"
                            in some_json_graph["graph"].keys()
                        ):
                            some_json_graph["graph"].pop("completed_stages")
                    input_graph = nx.node_link_graph(some_json_graph)

                # Only store unique graphs (overwrite the duplicate) with
                # identical ismorphic hash at the key.
                input_graphs[isomorphic_hash] = input_graph
    print(f"Found:{len(input_graphs.items())} unique input graphs.")

    # Sort the graphs to ensure it always returns the same order of input
    # graphs.
    # sorted_input_graphs: List[nx.Graph] = []
    # for some_hash in sorted(list(input_graphs.keys())):
    # sorted_input_graphs.append(input_graphs[some_hash])
    return input_graphs
