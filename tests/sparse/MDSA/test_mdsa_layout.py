"""Tests whether MDSA algorithm specification detects invalid
specifications."""
# pylint: disable=R0801
import unittest
from typing import List

import numpy as np
from snnbackends.networkx.LIF_neuron import Identifier
from typeguard import typechecked

from snnalgorithms.sparse.MDSA.layout import get_node_position


class Test_mdsa(unittest.TestCase):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_spike_once_values(self) -> None:
        """Verifies coordinates of spike_once nodes."""

        redundancy = 3
        graph_size = 3

        # expected_x = np.zeros((3, 4))
        expected_xy = np.zeros((3, 3, 2))
        actual_xy = np.zeros((3, 3, 2))

        # node_index, redundancy
        expected_xy[0, 0] = [0, 0]
        expected_xy[0, 1] = [220, 220]
        expected_xy[0, 2] = [440, 440]

        expected_xy[1, 0] = [0, 440.2]
        expected_xy[1, 1] = [220, 660.2]
        expected_xy[1, 2] = [440, 880.2]

        expected_xy[2, 0] = [0, 880.4]
        expected_xy[2, 1] = [220, 1100.4]
        expected_xy[2, 2] = [440, 1320.4]

        for i, node_index in enumerate(range(0, graph_size)):
            for j, node_redundancy in enumerate(range(0, redundancy)):
                identifiers: List = [
                    Identifier(
                        description="node_index",
                        position=0,
                        value=node_index,
                    )
                ]
                actual_xy[i, j] = get_node_position(
                    graph_size,
                    node_name="spike_once",
                    identifiers=identifiers,
                    node_redundancy=node_redundancy,
                )

                print(f"node_index={i},redundancy={j}")
                print(actual_xy[i, j])
                print(expected_xy[i, j])

                self.assertEqual(expected_xy[i, j, 0], actual_xy[i, j, 0])
                self.assertEqual(expected_xy[i, j, 1], actual_xy[i, j, 1])
