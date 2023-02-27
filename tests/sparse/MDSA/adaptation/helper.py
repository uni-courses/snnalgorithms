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
    max_redundancy: int,
    run_config: Run_config,
    test_object: Any,
) -> None:
    """Verifies the dead neuron spikes are taken over by the redundant
    neurons."""
    for dead_neuron_name in dead_neuron_names:
        # Loop through adapted graph to find spike times of dead neurons.
        for original_node_name in graphs_dict["snn_algo_graph"].nodes():
            # Get redundant neuron names.
            red_neuron_names: List[str] = get_redundant_neuron_names(
                max_redundancy=max_redundancy,
                original_node_name=original_node_name,
            )

            # If there is any redundant neuron that is not dead.
            if original_node_name == dead_neuron_name and any(
                red_neuron_name not in dead_neuron_names
                for red_neuron_name in red_neuron_names
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
                    dead_neuron_name=dead_neuron_name,
                    graphs_dict=graphs_dict,
                    output_config=output_config,
                    red_neuron_names=red_neuron_names,
                    max_redundancy=max_redundancy,
                    run_config=run_config,
                    spike_times=spike_times,
                    test_object=test_object,
                )


@typechecked
def get_redundant_neuron_names(
    max_redundancy: int, original_node_name: str
) -> List[str]:
    """Returns the names of the redundant neurons of a node."""
    redundant_neuron_names: List[str] = []
    for redundancy in range(1, max_redundancy):
        redundant_neuron_names.append(f"r_{redundancy}_{original_node_name}")
    return redundant_neuron_names


@typechecked
def perform_verification_for_each_spike_time(
    *,
    dead_neuron_name: str,
    graphs_dict: Dict,
    output_config: Output_config,
    red_neuron_names: List[str],
    max_redundancy: int,
    run_config: Run_config,
    spike_times: List[int],
    test_object: Any,
) -> None:
    """Performs the adaptation verification per timestep on which an adapted
    neuron fires."""

    # Verify the redundant neuron spikes <redundant> timesteps
    # after t.
    for t in spike_times:
        if not adapted_neuron_has_taken_over(
            dead_neuron_name=dead_neuron_name,
            graphs_dict=graphs_dict,
            max_redundancy=max_redundancy,
            red_neuron_names=red_neuron_names,
            t=t,
        ):
            print(
                f"Error, t={t}, red={range(1,max_redundancy)}: node:"
                + f"{red_neuron_names} does not take over."
            )
            run_config_filename = run_config_to_filename(
                run_config_dict=run_config.__dict__
            )

            create_svg_plot(
                run_config_filename=run_config_filename,
                graph_names=["rad_adapted_snn_graph"],
                graphs=graphs_dict,
                output_config=output_config,
            )
            test_object.assertTrue(False)


@typechecked
def adapted_neuron_has_taken_over(
    dead_neuron_name: str,
    graphs_dict: Dict,
    max_redundancy: int,
    red_neuron_names: List[str],
    t: int,
) -> bool:
    """Returns True if an adaptive neuron has taken over, False otherwise.

    TODO: write test for method.
    """
    for redundancy in range(1, max_redundancy):
        redundant_node_name = f"r_{redundancy}_{dead_neuron_name}"
        if redundant_node_name not in red_neuron_names:
            raise ValueError(
                f"Error, {redundant_node_name} not in:{redundant_node_name}"
            )

        if (
            graphs_dict["rad_adapted_snn_graph"]
            .nodes[redundant_node_name]["nx_lif"][t + redundancy]
            .spikes
        ):
            return True
    return False
