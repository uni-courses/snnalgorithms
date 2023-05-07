"""Contains the list of graphs that are used for radiation testing."""
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import customshowme
import networkx as nx
from snncompare.exp_config import Exp_config
from snncompare.graph_generation.export_input_graphs import (
    get_input_graph_output_filepath,
    has_outputted_input_graph_for_graph_size_and_nr,
    output_input_graph_if_not_exist,
)
from snncompare.import_results.helper import get_isomorphic_graph_hash
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.SNN_initialisation_properties import (
    SNN_initialisation_properties,
)


@customshowme.time
def create_mdsa_input_graphs_from_exp_config(
    exp_config: Exp_config,
) -> None:
    """Finds the maximum number of graphs per input size, for the MDSA
    algorithm and creates that many unique input graphs.

    Then outputs these.
    """

    if "MDSA" in exp_config.algorithms.keys():
        for graph_size, nr_of_graphs in exp_config.size_and_max_graphs:
            if not has_outputted_input_graph_for_graph_size_and_nr(
                graph_size=graph_size, graph_nr=nr_of_graphs - 1
            ):
                input_graphs: Dict[str, nx.Graph] = generate_mdsa_input_graphs(
                    graph_size=graph_size,
                    max_nr_of_graphs=nr_of_graphs,
                    seeds=exp_config.seeds,
                )
                for input_graph in input_graphs.values():
                    output_input_graph_if_not_exist(input_graph=input_graph)
    else:
        raise NotImplementedError("Error, algorithm not (yet) supported.")


@typechecked
def generate_mdsa_input_graphs(
    *,
    graph_size: int,
    max_nr_of_graphs: int,
    seeds: List[int],
) -> Dict[str, nx.Graph]:
    """Removes graphs that are not used, because of a maximum nr of graphs that
    is to be evaluated.

    TODO: export the input graphs to a pickle.
    Use the experiment config to generate the minimum number of required input
     graphs per graph size.
    """
    # Generate the input graphs.
    cum_input_graphs: Dict[str, nx.Graph] = {}
    for seed in seeds:
        # TODO: duplicate seeds are explored this way in
        # get_rand_planar_triangle_free_graph, if the seeds are consequitive.
        # This is because a new psuedo-random seed is created with
        # seed+graph_nr.
        # TODO: rewrite to only pass unique seeds.
        input_graphs: Dict[
            str, nx.Graph
        ] = get_rand_planar_triangle_free_graph(
            density_cutoff=0.01,
            max_nr_of_graphs=max_nr_of_graphs + 1,
            seed=seed,
            size=graph_size,
        )
        for input_graph_hash, input_graph in input_graphs.items():
            # pylint: disable=C0201
            if input_graph_hash not in cum_input_graphs.keys():
                cum_input_graphs[input_graph_hash] = input_graph

        if len(cum_input_graphs.values()) >= max_nr_of_graphs:
            break

    if len(cum_input_graphs.values()) < max_nr_of_graphs:
        raise ValueError(
            f"For input_graph of size:{graph_size}, I found:"
            + f"{len(cum_input_graphs)} graphs, yet expected graph_nr:"
            + f"{max_nr_of_graphs}. Please lower the max_graphs setting in:"
            + "size_and_max_graphs in the experiment configuration."
        )
    return input_graphs


def add_mdsa_initialisation_properties_to_input_graph(
    input_graph: nx.Graph, seed: int
) -> None:
    """Adds the initialisation properties into an input graph."""

    # Add the algorithm properties for the MDSA algorithm into the
    # input graphs as a dictionary. These properties are: the random
    # numbers that are used for the graph initialisation.
    if "alg_props" not in input_graph.graph.keys():
        input_graph.graph["alg_props"] = SNN_initialisation_properties(
            input_graph, seed
        ).__dict__

    if not isinstance(input_graph, nx.Graph):
        raise TypeError(
            "Error, the input graph is not a networkx graph:"
            + f"{type(input_graph)}"
        )


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
