"""Helps with the redundancy tests."""
import copy
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from snncompare.exp_config.Exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.export_plots.Plot_config import get_default_plot_config
from snncompare.export_results.helper import run_config_to_filename
from snncompare.optional_config.Output_config import (
    Extra_storing_config,
    Hover_info,
    Output_config,
    Zoom,
)
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnalgorithms.sparse.MDSA.get_results import get_neumann_results


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
    print(f"dead_neuron_names={dead_neuron_names}")
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
    *, max_redundancy: int, original_node_name: str
) -> List[str]:
    """Returns the names of the redundant neurons of a node."""
    redundant_neuron_names: List[str] = []
    for redundancy in range(1, max_redundancy + 1):
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
            print(f"Error, t={t}, red={range(1,max_redundancy)}: nodes:")
            print(red_neuron_names)
            print(f"don't take over from:{dead_neuron_name}.")
            run_config_filename = run_config_to_filename(
                run_config_dict=run_config.__dict__
            )

            create_svg_plot(
                run_config_filename=run_config_filename,
                graph_names=["adapted_snn_graph", "rad_adapted_snn_graph"],
                graphs=graphs_dict,
                output_config=output_config,
                run_config=run_config,
            )
            test_object.assertTrue(False)


@typechecked
def adapted_neuron_has_taken_over(
    *,
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
            len(
                graphs_dict["rad_adapted_snn_graph"].nodes[
                    redundant_node_name
                ]["nx_lif"]
            )
            <= t + redundancy
        ):
            nr_of_timesteps = len(
                graphs_dict["rad_adapted_snn_graph"].nodes[
                    redundant_node_name
                ]["nx_lif"]
            )
            print(f"len rad_adapted_snn_graph={nr_of_timesteps}")
            print("Error, len rad_adapted_snn_graph was not enough.")
            print(f"t={t}, redundancy={redundancy}")
            return False
        if (
            graphs_dict["rad_adapted_snn_graph"]
            .nodes[redundant_node_name]["nx_lif"][t + redundancy]
            .spikes
        ):
            return True
    return False


@typechecked
def create_default_hover_info(*, exp_config: Exp_config) -> Hover_info:
    """Create duplicate Hover_info that is used to generate the data belonging
    to each run config, using the Experiment runner."""

    hover_info = Hover_info(
        incoming_synapses=True,
        neuron_models=exp_config.neuron_models,
        # pylint: disable=R0801
        neuron_properties=[
            "spikes",
            "a_in",
            "bias",
            "du",
            "u",
            "dv",
            "v",
            "vth",
        ],
        node_names=True,
        outgoing_synapses=True,
        synaptic_models=exp_config.synaptic_models,
        synapse_properties=["weight"],
    )
    return hover_info


@typechecked
def create_default_output_config(*, exp_config: Exp_config) -> Output_config:
    """Create duplicate Output_config that is used to generate the data
    belonging to each run config, using the Experiment runner."""
    output_config = Output_config(
        recreate_stages=[1, 2, 4],
        export_types=[],
        zoom=Zoom(
            create_zoomed_image=False,
            left_right=None,
            bottom_top=None,
        ),
        output_json_stages=[1, 2, 4],
        hover_info=create_default_hover_info(exp_config=exp_config),
        extra_storing_config=Extra_storing_config(
            count_spikes=False,
            count_neurons=False,
            count_synapses=False,
            skip_stage_2_output=True,
            show_images=True,
            store_died_neurons=False,
        ),
    )
    return output_config


@typechecked
def long_exp_config_for_mdsa_testing_with_adaptation() -> Exp_config:
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    long_mdsa_testing: Dict = {
        "adaptations": {"redundancy": [2, 4]},
        "algorithms": {
            "MDSA": get_algo_configs(
                algo_spec=MDSA(list(range(0, 6, 1))).__dict__
            )
        },
        # TODO: Change into list with "Seeds"
        "seeds": [7],
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 2,
        "min_graph_size": 3,
        "max_graph_size": 5,
        "size_and_max_graphs": [(3, 1), (4, 3), (5, 6)],
        # Move into "overwrite options"
        "radiations": {},
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }

    # The ** loads the dict into the object.
    exp_config = Exp_config(**long_mdsa_testing)
    return exp_config


@typechecked
def get_dead_neuron_names(
    *,
    redundancy_levels: List[int],
    snn_algo_graph: nx.DiGraph,
) -> List[List[str]]:
    """Returns lists with a single dead neuron name (one for each original snn
    neuron)."""
    combinations: List[List[str]] = []

    for node_name in snn_algo_graph.nodes():
        node_names: List[str] = []
        if 0 in redundancy_levels:
            node_names.append(node_name)
        for red_level in redundancy_levels:
            if red_level > 0:
                node_names.append(f"r_{red_level}_{node_name}")
        combinations.append(node_names)
    return combinations


@typechecked
def assert_run_config_json_results(
    *,
    test_object: Any,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph]],
    run_config: Run_config,
) -> None:
    """Verifies the results of a run config using the json result output.

    TODO: update expected_stages.
    """

    # Verify results are as expected.
    expected_node_names: Dict[str, int] = get_neumann_results(
        input_graph=graphs_dict["input_graph"],
        m_val=run_config.algorithm["MDSA"]["m_val"],
        rand_props=graphs_dict["input_graph"].graph["alg_props"],
        seed=run_config.seed,
        size=run_config.graph_size,
    )

    for graph_name, snn_graph in graphs_dict.items():
        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:
            # Verify the SNN graphs have completed simulation stage 2.
            test_object.assertTrue(
                # pylint: disable=R1733
                2
                in graphs_dict[graph_name].graph["completed_stages"]
            )

            actual_node_names: Dict[str, int] = snn_graph.graph["results"]

            # Remove the passed boolean, and redo results verification.
            copy_actual_node_names = copy.deepcopy(actual_node_names)
            copy_actual_node_names.pop("passed")

            # Verify node names are identical.
            test_object.assertEquals(
                copy_actual_node_names.keys(), expected_node_names.keys()
            )

            # Verify the expected nodes are the same as the actual nodes.
            for key, expected_val in expected_node_names.items():
                test_object.assertEquals(
                    expected_val, copy_actual_node_names[key]
                )

            test_object.assertTrue(actual_node_names["passed"])


@typechecked
def get_run_config_and_results_dicts_for_large_test_scope(
    *, with_adaptation_only: bool, min_red_level: Optional[int] = None
) -> Tuple[Dict[Run_config, Dict], Output_config]:
    """Sets up the long mdsa test scope for MDSA redundancy testing."""
    mdsa_settings: Exp_config = (
        long_exp_config_for_mdsa_testing_with_adaptation()
    )
    print("")
    print("mdsa_settings=")
    pprint(mdsa_settings.__dict__)

    # Generate test scope.
    mdsa_settings.export_types = None
    output_config: Output_config = create_default_output_config(
        exp_config=mdsa_settings
    )

    # Generate run_configs
    full_exp_runner = Experiment_runner(
        exp_config=mdsa_settings,
        output_config=output_config,
        reverse=True,
        perform_run=False,
        specific_run_config=None,
    )

    run_config_results: Dict[Run_config, Dict] = {}
    for run_config in full_exp_runner.run_configs:
        # If you want to only test a specific run.
        # if run_config.unique_id == (
        # "962d5dee640f590fa7d1b85c2e220567f2c1851a981ebc1bd6463d0fe79d3a50"
        # ):
        # Only test run_configs with adaptation if desired.
        if not with_adaptation_only or (
            # If the run_config has adaptation, and no minimum redundancy
            # level is required, add it.
            list(run_config.adaptation.keys()) == ["redundancy"]
            and (
                min_red_level is None
                # If the run_config has adaptation, and the redundancy
                # level is equal to, or larger than required minimum
                # redundancy, add it.
                or run_config.adaptation["redundancy"] >= min_red_level
            )
        ):
            # Get the original results dict to manually execute experiment.
            original_results_nx_graphs: Dict = (
                full_exp_runner.perform_run_stage_1(
                    exp_config=mdsa_settings,
                    output_config=output_config,
                    plot_config=get_default_plot_config(),
                    run_config=run_config,
                )
            )
            run_config_results[run_config] = original_results_nx_graphs
    return run_config_results, output_config


@typechecked
def get_spike_window_per_neuron_type(
    *,
    t: int,
    max_redundancy: int,
) -> List[int]:
    """Returns the timesteps at which an original neuron-, or redundant neuron
    of a specific type may fire."""
    time_window: List[int] = []
    for i in range(t, t + max_redundancy + 1):
        time_window.append(i)
    return time_window
