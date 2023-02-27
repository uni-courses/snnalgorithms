"""Checks when a dead neurons spikes in the non-radiated adapted SNN.

Then asserts the redundant `n` neuron spikes `n` timesteps after the
neuron would spike in the unradiated version.
"""


from typing import Any, Dict, List, Union

import networkx as nx
from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.export_results.helper import run_config_to_filename
from snncompare.optional_config import Output_config
from snncompare.run_config import Run_config
from typeguard import typechecked


@typechecked
def assert_redundant_neuron_takes_over(
    *,
    dead_neuron_names: List[str],
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph]],
    output_config: Output_config,
    redundancy: int,
    run_config: Run_config,
    test_object: Any,
) -> None:
    """Verifies the dead neuron spikes are taken over by the redundant
    neurons."""
    for dead_neuron_name in dead_neuron_names:
        # Loop through adapted graph to find spike times of dead neurons.
        for adapted_nodename in graphs_dict["adapted_snn_graph"].nodes():
            # Get redundant neuron name.
            redundant_node_name = f"r_{redundancy}_{dead_neuron_name}"
            if (
                adapted_nodename == dead_neuron_name
                and redundant_node_name not in dead_neuron_names
            ):
                spike_times: List[int] = []
                # Get the time at which the redundant neuron should spike.
                for t, adapted_nx_lif in enumerate(
                    graphs_dict["adapted_snn_graph"].nodes[dead_neuron_name][
                        "nx_lif"
                    ]
                ):
                    if adapted_nx_lif.spikes:
                        spike_times.append(t)
                perform_verification_for_each_spike_time(
                    graphs_dict=graphs_dict,
                    output_config=output_config,
                    redundant_node_name=redundant_node_name,
                    redundancy=redundancy,
                    run_config=run_config,
                    spike_times=spike_times,
                    test_object=test_object,
                )


@typechecked
def perform_verification_for_each_spike_time(
    *,
    graphs_dict: Dict,
    output_config: Output_config,
    redundant_node_name: str,
    redundancy: int,
    run_config: Run_config,
    spike_times: List[int],
    test_object: Any,
) -> None:
    """Performs the adaptation verification per timestep on which an adapted
    neuron fires."""

    print(
        "rad adap nodes=" + f'{graphs_dict["rad_adapted_snn_graph"].nodes()}'
    )
    # Verify the redundant neuron spikes <redundant> timesteps
    # after t.
    for t in spike_times:
        if (
            not graphs_dict["rad_adapted_snn_graph"]
            .nodes[redundant_node_name]["nx_lif"][t + redundancy]
            .spikes
        ):
            print(f"{t}, red={redundancy}:{redundant_node_name}")
            run_config_filename = run_config_to_filename(
                run_config_dict=run_config.__dict__
            )

            create_svg_plot(
                run_config_filename=run_config_filename,
                graph_names=["rad_adapted_snn_graph"],
                graphs=graphs_dict,
                output_config=output_config,
            )
        test_object.assertTrue(
            graphs_dict["rad_adapted_snn_graph"]
            .nodes[redundant_node_name]["nx_lif"][t + redundancy]
            .spikes
        )
