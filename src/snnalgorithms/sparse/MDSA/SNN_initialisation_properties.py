"""Takes an input graph and generates an SNN that solves the MDSA algorithm by
Alipour et al."""
from typing import List

import networkx as nx
from snncompare.helper import generate_list_of_n_random_nrs
from typeguard import typechecked


class SNN_initialisation_properties:
    """Contains the properties required to compute Alipour algorithm
    results."""

    @typechecked
    def __init__(self, G: nx.Graph, seed: int) -> None:
        # Initialise properties for Alipour algorithm
        rand_ceil = self.get_random_ceiling(G)
        rand_nrs = generate_list_of_n_random_nrs(
            G=G, max_val=rand_ceil, seed=seed
        )
        # Store properties in object.
        self.rand_ceil = rand_ceil
        self.rand_nrs = rand_nrs
        self.rand_edge_weights = self.get_rand__degree_receiver_edge_weights(
            G, rand_ceil, rand_nrs
        )

    @typechecked
    def get_random_ceiling(self, G: nx.Graph) -> int:
        """Generate the maximum random ceiling.

        +2 to allow selecting a larger range of numbers than the number
        of # nodes in the graph.

        :param G: The original graph on which the MDSA algorithm is ran.
        """
        rand_ceil = len(G) - 1
        return rand_ceil

    @typechecked
    def get_degree_receiver_offset(self, G: nx.Graph, rand_ceil: int) -> int:
        """Compute offset to rand_nrs to ensure the degree_receiver current
        u[1] always starts negative. That is important because otherwise
        degree_receiver neurons in their WTA circuits might spike
        simultaneously on t=2.

        The a_in of the degree_receiver_x_y neuron can be described as:
            + a spike from at most n-1 spike_once neurons, with weight=n-1.
            + a random nr in range/of weight: [0, n-1] (or [0, rand_ceil-1]).
            + selector_excitation of weight +1
            - this offset.
        So at most the degree_receivers get: (n-1)*(n-1)+(n-1) as positive
        input at the start.
        """
        spike_once_offset = (len(G) - 1) * (len(G) - 1)
        rand_nr_offset = rand_ceil
        degree_receiver_offset = -spike_once_offset - rand_nr_offset
        return degree_receiver_offset

    @typechecked
    def get_rand__degree_receiver_edge_weights(
        self, G: nx.Graph, rand_ceil: int, rand_nrs: List[int]
    ) -> List[int]:
        """Returns the list with random initial synaptic weights for the rand_
        neurons."""
        degree_receiver_offset = self.get_degree_receiver_offset(G, rand_ceil)
        rand_edge_weights = [degree_receiver_offset - x for x in rand_nrs]
        return rand_edge_weights
