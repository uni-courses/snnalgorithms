"""Takes an input graph and generates an SNN that solves the MDSA algorithm by
Alipour et al."""
from typing import List

from networkx.classes.graph import Graph
from snncompare.helper import generate_list_of_n_random_nrs
from typeguard import typechecked


class SNN_initialisation_properties:
    """Contains the properties required to compute Alipour algorithm
    results."""

    @typechecked
    def __init__(self, G: Graph, seed: int) -> None:

        # Initialise properties for Alipour algorithm
        rand_ceil = self.get_random_ceiling(G)
        rand_nrs = generate_list_of_n_random_nrs(
            G, max_val=rand_ceil, seed=seed
        )
        delta = self.get_delta()
        spread_rand_nrs = self.spread_rand_nrs_with_delta(delta, rand_nrs)
        inhibition = self.get_inhibition(delta, G, rand_ceil)
        initial_rand_current = self.get_initial_random_current(
            inhibition, spread_rand_nrs
        )

        # Store properties in object.
        self.rand_ceil = rand_ceil
        self.rand_nrs = rand_nrs
        self.delta = delta
        self.spread_rand_nrs = spread_rand_nrs
        self.inhibition = inhibition
        self.initial_rand_current = initial_rand_current

    @typechecked
    def get_random_ceiling(self, G: Graph) -> int:
        """Generate the maximum random ceiling.

        +2 to allow selecting a larger range of numbers than the number
        of # nodes in the graph.

        :param G: The original graph on which the MDSA algorithm is ran.
        """
        rand_ceil = len(G) + 0
        return rand_ceil

    @typechecked
    def get_delta(self) -> int:
        """Make the random numbers differ with at least delta>=2.

        This is to prevent multiple degree_receiver_x_y neurons (that
        differ less than delta) in a single WTA circuit to spike before
        they are inhibited by the first winner. This inhibition goes via
        the selector neuron and has a delay of 2. So a winner should
        have a difference of at least 2.
        """
        delta = 1
        return delta

    @typechecked
    def spread_rand_nrs_with_delta(
        self, delta: int, rand_nrs: List[int]
    ) -> List[int]:
        """Spread the random numbers with delta to ensure 1 winner in WTA
        circuit.

        :param delta: Value of how far the rand_nrs are separated.
        :param rand_nrs: List of random numbers that are used.
        """
        spread_rand_nrs = [x * delta for x in rand_nrs]
        return spread_rand_nrs

    @typechecked
    def get_inhibition(self, delta: int, G: Graph, rand_ceil: int) -> int:
        """Add inhibition to rand_nrs to ensure the degree_receiver current
        u[1] always starts negative. The a_in of the degree_receiver_x_y neuron
        is.

        : the incoming spike_once_x weights+rand_x neurons+selector_excitation
        - There are at most n incoming spike signals.
        - Each spike_once should have a weight of at least random_ceiling+1.
        That is because the random value should map to 0<rand<1 with respect
        to the difference of 1 spike_once more or less.
        - The random_ceiling is specified.
        - The excitatory neuron comes in at +1, a buffer of 1 yields+2.
        Hence, the inhibition is computed as:

        :param delta: Value of how far the rand_nrs are separated. param G:
        :param rand_ceil: Ceiling of the range in which rand nrs can be
        generated.
        :param G: The original graph on which the MDSA algorithm is ran.
        """
        inhibition = len(G) * (rand_ceil * delta + 1) + (rand_ceil) * delta + 1
        return inhibition

    @typechecked
    def get_initial_random_current(
        self, inhibition: int, rand_nrs: List[int]
    ) -> List[int]:
        """Returns the list with random initial currents for the rand_ neurons.

        :param inhibition: Value of shift of rand_nrs to ensure
        degree_receivers start at negative current u[t-0].
        :param rand_nrs: List of random numbers that are used.
        """
        initial_rand_current = [x - inhibition for x in rand_nrs]
        return initial_rand_current
