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

    # TODO: change into dict, and if selector neuron, check only if
    # two different redundant neurons spike, not whether more than one spike
    # has been registered within the time_window.
    #
    # This should not be necessary, because the r_2_selector_0_0 should not
    # spike at t=3, if the selector_0_0 spikes at t=2. It should spike for the
    # first time at t=4. To inspect this, see manually whether the
    # r_2_selector_0_0 spikes at t=3 (simultaneously with r_1_selector_0_0) if
    # only selector_0 dies of radiation.
    # 962d5dee640f590fa7d1b85c2e220567f2c1851a981ebc1bd6463d0fe79d3a50
    nr_of_redundant_neuron_spikes: int = 0
    for redundant_node_name in redundant_node_names:
        for timestep in timestep_window:
            if (
                len(rad_adapted_graph.nodes[redundant_node_name]["nx_lif"])
                > timestep
            ):
                if rad_adapted_graph.nodes[redundant_node_name]["nx_lif"][
                    timestep
                ].spikes:
                    nr_of_redundant_neuron_spikes += 1
            else:
                # The network is already done, so the redundant neurons cannot
                # spike anymore within the remaining timestep_window.
                break
    if nr_of_redundant_neuron_spikes != 1:
        print(f"timestep_window={timestep_window}")
        print(f"redundant_node_names={redundant_node_names}")
        print(f"nr_of_redundant_neuron_spikes={nr_of_redundant_neuron_spikes}")
        return False
    return True
