"""Contains the specification of and maximum values of the algorithm
settings."""
import sys

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

        # Specify supported values for u
        self.u_min: float = -10
        self.u_max: float = 10
        self.u_len: int = 20

        # Specify supported values for du.
        self.du_min: float = -1
        self.du_max: float = 1
        self.du_len: int = 10

        # Specify svpported valves for v
        self.v_min: float = -10
        self.v_max: float = 10
        self.v_len: int = 20

        # Specify svpported valves for dv.
        self.dv_min: float = -1
        self.dv_max: float = 1
        self.dv_len: int = 10

        # Specify supported biasalbiases for bias
        self.bias_min: float = -10
        self.bias_max: float = 10
        self.bias_len: int = 20

        # Specify supported vthalvthes for vth
        self.vth_min: float = -10
        self.vth_max: float = 10
        self.vth_len: int = 20

        # Specify supported vthalvthes for vth
        self.weight_min: int = -20
        self.weight_max: int = 20


# pylint: disable=R0903
# pylint: disable=R0801
class Discovery_algo:
    """Create a particular configuration for the neuron Discovery algorithm."""

    @typechecked
    def __init__(self, disco: Discovery) -> None:
        max_time: int = 10000
        count = 0
        total = (
            disco.du_len
            * disco.dv_len
            * disco.vth_len
            * disco.bias_len
            * (disco.weight_max - disco.weight_min)
        )

        # pylint: disable=R1702
        for du in [
            disco.du_min + i * (disco.du_max - disco.du_min) / disco.du_len
            for i in range(disco.du_len)
        ]:
            for dv in [
                disco.dv_min + i * (disco.dv_max - disco.dv_min) / disco.dv_len
                for i in range(disco.dv_len)
            ]:
                for vth in [
                    disco.vth_min
                    + i * (disco.vth_max - disco.vth_min) / disco.vth_len
                    for i in range(disco.vth_len)
                ]:
                    for bias in [
                        disco.bias_min
                        + i
                        * (disco.bias_max - disco.bias_min)
                        / disco.bias_len
                        for i in range(disco.bias_len)
                    ]:
                        for weight in range(
                            disco.weight_min, disco.weight_max
                        ):
                            # Create neuron.
                            lif_neuron = LIF_neuron(
                                name="",
                                bias=bias,
                                du=du,
                                dv=dv,
                                vth=vth,
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
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write(f"[ {progress} ] {percent * 100:.2f}%")
        sys.stdout.flush()
