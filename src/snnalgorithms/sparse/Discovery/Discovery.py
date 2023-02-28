"""Contains the specification of and maximum values of the algorithm
settings."""
import sys
from typing import List, Optional, Union

import networkx as nx
from snnbackends.networkx.LIF_neuron import (
    LIF_neuron,
    Synapse,
    print_neuron_properties_per_graph,
)
from snnbackends.networkx.run_on_networkx import (
    create_neuron_for_next_timestep,
    run_simulation_with_networkx_for_1_timestep,
)
from snnbackends.verify_graph_is_snn import verify_networkx_snn_spec
from typeguard import typechecked


# pylint: disable=R0902
# pylint: disable=R0903
class Discovery:
    """Specification of algorithm specification. Algorithm: Minimum Dominating
    Set Approximation by Alipour.

    Example usage: default_MDSA_alg=MDSA(some_vals=list(range(0, 4, 1)))
    """

    @typechecked
    def __init__(
        self,
    ) -> None:
        self.name = "Discovery"

        # Specify supported values for du.
        du_min: float = -1
        du_max: float = 1
        du_len: int = 10
        self.du_range = self.get_range(
            min_val=du_min,
            max_val=du_max,
            length=du_len,
        )

        # Specify svpported valves for dv.
        dv_min: float = -1
        dv_max: float = 1
        dv_len: int = 10
        self.dv_range = self.get_range(
            min_val=dv_min,
            max_val=dv_max,
            length=dv_len,
        )

        # Specify supported biasalbiases for bias
        bias_min: float = -10
        bias_max: float = 10
        bias_len: int = 20
        self.bias_range = self.get_range(
            min_val=bias_min,
            max_val=bias_max,
            length=bias_len,
        )

        # Specify supported vthalvthes for vth
        vth_min: float = -10
        vth_max: float = 10
        vth_len: int = 20
        self.vth_range = self.get_range(
            min_val=vth_min,
            max_val=vth_max,
            length=vth_len,
        )

        # Specify supported vthalvthes for vth
        weight_min: int = -20
        weight_max: int = 20
        self.weight_range = self.get_range(
            min_val=weight_min,
            max_val=weight_max,
        )

    @typechecked
    def get_range(
        self,
        min_val: Union[float, int],
        max_val: Union[float, int],
        length: Optional[int] = None,
    ) -> List[Union[float, int]]:
        """Returns a list with the values in a range."""
        if length is not None:
            return [
                min_val + i * (max_val - min_val) / length
                for i in range(length)
            ]
        if not isinstance(min_val, int) or not isinstance(min_val, int):
            raise TypeError(
                "Error, min or max value without length specification "
                + f"requires integers. Found min:{min_val},max:{max_val}"
            )
        return list(range(int(min_val), int(max_val)))


class DiscoveryRanges(Discovery):
    """Specification of algorithm specification. Algorithm: Minimum Dominating
    Set Approximation by Alipour.

    Example usage: default_MDSA_alg=MDSA(some_vals=list(range(0, 4, 1)))
    """

    @typechecked
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.name = "Discovery"

        # Specify supported values for du.
        self.du_range = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]

        # Specify supported values for dv.
        self.dv_range = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]

        # Specify supported values for u
        self.bias_range = list(range(-10, 10))

        # Specify supported values for u
        self.vth_range = list(range(-10, 10))

        # Specify supported values for weight
        self.weight_range = list(range(-10, 10))


# pylint: disable=R0903
# pylint: disable=R0801
class Discovery_algo:
    """Create a particular configuration for the neuron Discovery algorithm."""

    @typechecked
    def __init__(self, disco: Discovery) -> None:
        max_time: int = 10000
        count = 0
        total = (
            len(disco.du_range)
            * len(disco.dv_range)
            * len(disco.vth_range)
            * len(disco.bias_range)
            * len(disco.weight_range)
        )
        print(f"du:{disco.du_range}")
        print(f"dv:{disco.dv_range}")
        print(f"bias:{disco.bias_range}")
        print(f"vth:{disco.vth_range}")
        print(f"weight:{disco.weight_range}")

        # pylint: disable=R1702
        for du in disco.du_range:
            for dv in disco.dv_range:
                for bias in disco.bias_range:
                    for vth in disco.vth_range:
                        for weight in disco.weight_range:
                            # Create neuron.
                            lif_neuron = LIF_neuron(
                                name="",
                                bias=float(bias),
                                du=float(du),
                                dv=float(dv),
                                vth=float(vth),
                            )
                            count = count + 1
                            self.drawProgressBar(
                                percent=count / total, barLen=100
                            )

                            if self.is_expected_neuron_I(
                                lif_neuron=lif_neuron,
                                max_time=max_time,
                                weight=weight,
                            ):
                                print(
                                    f"du={du},dv={dv},vth={vth},bias={bias},"
                                    + f"weight={weight}"
                                )
                                print("FOUND")
                                sys.exit()

    def is_expected_neuron_I(
        self, lif_neuron: LIF_neuron, max_time: int, weight: int
    ) -> bool:
        """Determines whether a neuron is of type I.

        Type I is arbitrarily defined as: 'does not spike for 2
        timesteps, and then spikes indefinitely.'. (Because I would like
        to use such a neuron.).
        """

        snn_graph = nx.DiGraph()
        node_name: str = "0"
        snn_graph.add_nodes_from(
            [node_name],
        )
        snn_graph.nodes[node_name]["nx_lif"] = [lif_neuron]
        snn_graph.add_edges_from(
            [(node_name, node_name)],
            synapse=Synapse(
                weight=weight,
                delay=0,
                change_per_t=0,
            ),
        )

        # Simulate neuron for at most max_time timesteps, as long as it behaves
        # as desired.
        for t in range(0, max_time):
            # Copy the neurons into the new timestep.
            verify_networkx_snn_spec(snn_graph=snn_graph, t=t, backend="nx")
            create_neuron_for_next_timestep(snn_graph=snn_graph, t=t)

            verify_networkx_snn_spec(
                snn_graph=snn_graph, t=t + 1, backend="nx"
            )
            # Simulate neuron.
            run_simulation_with_networkx_for_1_timestep(
                snn_graph=snn_graph, t=t + 1
            )

            # If neuron behaves, continue, otherwise move on to next neuron.
            if snn_graph.nodes[node_name]["nx_lif"][
                t
            ].spikes != self.expected_spike_pattern_I(t):
                return False
            if not self.within_neuron_property_bounds(
                lif_neuron=snn_graph.nodes[node_name]["nx_lif"][t]
            ):
                return False

            if t > 100:
                print_neuron_properties_per_graph(
                    G=snn_graph, static=False, t=t, neuron_type="nx_lif"
                )

        return True

    @typechecked
    def expected_spike_pattern_I(self, t: int) -> bool:
        """Specifies the expected spike pattern for neuron type I."""
        if t < 3:
            return False
        return True

    @typechecked
    def within_neuron_property_bounds(self, lif_neuron: LIF_neuron) -> bool:
        """If the voltage exceeds 100, return False.."""
        if lif_neuron.v.get() > 100 or lif_neuron.v.get() < -100:
            return False
        if lif_neuron.u.get() > 100 or lif_neuron.u.get() < -100:
            return False
        return True

    @typechecked
    def drawProgressBar(self, percent: float, barLen: int = 20) -> None:
        """Draws a completion bar."""
        sys.stdout.write("'\r'")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write(f"[ {progress} ] {percent * 100:.2f}%")
        sys.stdout.flush()
