"""Checks when a dead neurons spikes in the non-radiated adapted SNN.

Then asserts the redundant `n` neuron spikes `n` timesteps after the
neuron would spike in the unradiated version.
"""

from typing import Dict, List, Union

import networkx as nx
from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.export_results.helper import run_config_to_filename
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked

from tests.sparse.MDSA.adaptation.redundancy_helper import (
    get_redundant_neuron_names,
    get_spike_window_per_neuron_type,
)


@typechecked
def assert_n_redundant_neuron_takes_over(
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph]],
    dead_neuron_names: List[str],
    max_redundancy: int,
    run_config: Run_config,
    output_config: Output_config,
) -> None:
    """Verifies the n dead neuron spikes are taken over by the redundant
    neurons.

    Also verifies only 1 redundant neuron spikes per spike window.
    """

    # For each spike in the adapted network, assert no redundant neuron spikes.
    adapted_graph: nx.DiGraph = graphs_dict["adapted_snn_graph"]
    rad_adapted_graph: nx.DiGraph = graphs_dict["rad_adapted_snn_graph"]
    snn_graph: nx.DiGraph = graphs_dict["snn_algo_graph"]

    # Per original node, check when it spikes in the adapted graph.
    # pylint: disable=R1702
    for node_name in snn_graph:
        if node_name in dead_neuron_names:
            if not any(
                ignored_node_name in node_name
                for ignored_node_name in ["connector", "counter", "terminator"]
            ):
                for t, adapted_nx_lif in enumerate(
                    adapted_graph.nodes[node_name]["nx_lif"]
                ):
                    if adapted_nx_lif.spikes:
                        # Assert no duplicate spikes exist,
                        if not redundant_neuron_takes_over_without_duplicates(
                            rad_adapted_graph=rad_adapted_graph,
                            max_redundancy=max_redundancy,
                            node_name=node_name,
                            t=t,
                        ):
                            print(
                                "Error, redundant neuron did not take over "
                                + f"correctly for:{dead_neuron_names} at {t}."
                            )

                            # Visualise the snn behaviour
                            run_config_filename = run_config_to_filename(
                                run_config_dict=run_config.__dict__
                            )

                            create_svg_plot(
                                run_config_filename=run_config_filename,
                                graph_names=[
                                    "adapted_snn_graph",
                                    "rad_adapted_snn_graph",
                                ],
                                graphs=graphs_dict,
                                output_config=output_config,
                            )
                            raise ValueError()


@typechecked
def redundant_neuron_takes_over_without_duplicates(
    *,
    rad_adapted_graph: nx.DiGraph,
    max_redundancy: int,
    node_name: str,
    t: int,
) -> bool:
    """Returns True if AT MOST 1 redundant neuron of a specific type spikes
    within the redundancy spike range of the original neuron.

    Returns False if multiple redundant neurons of a specific type spike
    within the redundancy spike range of the original neuron.

    Returns False, the redundant neurons of a specific type do not spike
    within the redundancy spike range of the original neuron.
    """

    # For each spike in the adapted network, assert no redundant neuron spikes.
    timestep_window: List[int] = get_spike_window_per_neuron_type(
        t=t,
        max_redundancy=max_redundancy,
    )

    redundant_node_names: List[str] = get_redundant_neuron_names(
        max_redundancy=max_redundancy,
        original_node_name=node_name,
    )

    nr_of_redundant_neuron_spikes = get_nr_of_redundant_neurons_spiking(
        rad_adapted_graph=rad_adapted_graph,
        redundant_node_names=redundant_node_names,
        timestep_window=timestep_window,
    )

    if nr_of_redundant_neuron_spikes != 1:
        print(f"timestep_window={timestep_window}")
        print(f"redundant_node_names={redundant_node_names}")
        print(f"nr_of_redundant_neuron_spikes={nr_of_redundant_neuron_spikes}")
        return False
    return True


@typechecked
def get_nr_of_redundant_neurons_spiking(
    redundant_node_names: List[str],
    timestep_window: List[int],
    rad_adapted_graph: nx.DiGraph,
) -> int:
    """Counts the nr of simultaneous redundant spikes within a timestep window,
    with the purpose of finding duplicate spikes. Since some neuron types
    continue firing, they may lead to multiple spikes within the same time
    window. For example the selector neuron. To prevent the, expected,
    continuous firing of such neurons to say two, different redundant neurons
    spiked within the same window, a dictionary with nr of spikes per redundant
    neuron is created. For continuously spiking neurons, the dictionary is
    checked for the nr of neurons that have spiked, instead of the nr of
    spikes.

    Returns the number of redundant neurons that spiked in the time
    window.
    """
    # Create dictionary with redundant spikes, one per redundant neuron.
    redundant_spikes: Dict[str, int] = {}
    for redundant_node_name in redundant_node_names:
        redundant_spikes[redundant_node_name] = 0

    # Create absolute neuron spike count.
    nr_of_redundant_neuron_spikes: int = 0

    # Count the nr of redundant spikes and store them.
    for redundant_node_name in redundant_node_names:
        for timestep in timestep_window:
            if (
                len(rad_adapted_graph.nodes[redundant_node_name]["nx_lif"])
                > timestep
            ):
                if rad_adapted_graph.nodes[redundant_node_name]["nx_lif"][
                    timestep
                ].spikes:
                    redundant_spikes[redundant_node_name] += 1
                    nr_of_redundant_neuron_spikes += 1
            else:
                # The network is already done, so the redundant neurons cannot
                # spike anymore within the remaining timestep_window.
                break
    for redundant_node_name in redundant_node_names:
        if "selector" in redundant_node_name:
            # Return nr of spikes as the nr of different redundant neurons that
            # have spiked.
            return sum(
                spike_count > 0 for spike_count in redundant_spikes.values()
            )
    return nr_of_redundant_neuron_spikes
