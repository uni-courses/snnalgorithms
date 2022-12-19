"""Creates the MDSA snn synapses."""

import networkx as nx
from snnbackends.networkx.LIF_neuron import Synapse
from snncompare.exp_setts.run_config.Run_config import Run_config
from typeguard import typechecked


@typechecked
def create_MDSA_recurrent_synapses(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    recurrent_weight: int,
    run_config: Run_config,
) -> nx.DiGraph:
    """Creates the synapses between the neurons for the MDSA algorithm."""

    # Create spike_once nodes.

    create_recurrent_spike_once_synapse(
        input_graph, mdsa_snn, recurrent_weight
    )

    create_recurrent_degree_receiver_synapse(
        input_graph,
        mdsa_snn,
        recurrent_weight,
        run_config,
    )

    # Create recurrent rand_synapses
    create_recurrent_rand_synapse(
        input_graph,
        mdsa_snn,
        recurrent_weight,
    )

    create_recurrent_next_round_synapse(
        mdsa_snn,
        run_config,
    )

    return mdsa_snn


@typechecked
def create_recurrent_spike_once_synapse(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    recurrent_weight: int,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    for node_index in input_graph.nodes:
        mdsa_snn.add_edges_from(
            [
                (
                    f"spike_once_{node_index}",
                    f"spike_once_{node_index}",
                )
            ],
            synapse=Synapse(
                weight=recurrent_weight,
                delay=0,
                change_per_t=0,
            ),
        )


# pylint: disable=R0913
@typechecked
def create_recurrent_degree_receiver_synapse(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    recurrent_weight: int,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    for node_index in input_graph.nodes:
        # pylint: disable=R0801
        for node_neighbour in nx.all_neighbors(input_graph, node_index):
            if node_index != node_neighbour:
                for m_val in range(
                    0, run_config.algorithm["MDSA"]["m_val"] + 1
                ):
                    mdsa_snn.add_edges_from(
                        [
                            (
                                f"degree_receiver_{node_index}_"
                                + f"{node_neighbour}_{m_val}",
                                f"degree_receiver_{node_index}_"
                                + f"{node_neighbour}_{m_val}",
                            )
                        ],
                        synapse=Synapse(
                            weight=recurrent_weight,
                            delay=0,
                            change_per_t=0,
                        ),
                    )


@typechecked
def create_recurrent_rand_synapse(
    input_graph: nx.Graph,
    mdsa_snn: nx.DiGraph,
    recurrent_weight: int,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    for node_index in input_graph.nodes:
        mdsa_snn.add_edges_from(
            [
                (
                    f"rand_{node_index}",
                    f"rand_{node_index}",
                )
            ],
            synapse=Synapse(
                weight=recurrent_weight,
                delay=0,
                change_per_t=0,
            ),
        )


@typechecked
def create_recurrent_next_round_synapse(
    mdsa_snn: nx.DiGraph,
    run_config: Run_config,
) -> None:
    """Creates the outgoing synapses for the connecting node in the MDSA
    algorithm."""
    # Create recurrent synapse
    for m_val in range(1, run_config.algorithm["MDSA"]["m_val"] + 1):
        mdsa_snn.add_edges_from(
            [
                (
                    f"next_round_{m_val}",
                    f"next_round_{m_val}",
                )
            ],
            # pylint: disable=R0801
            synapse=Synapse(
                weight=-5,  # TODO: why is this not "recurrent_weight"?
                delay=0,
                change_per_t=0,
            ),
        )
