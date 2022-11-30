"""Creates the MDSA snn."""


import networkx as nx
from snnbackends.networkx.LIF_neuron import (
    Identifier,
    LIF_neuron,
    Recurrent_synapse,
)
from typeguard import typechecked


@typechecked
def get_new_mdsa_graph(stage_1_nx_graphs: dict) -> nx.DiGraph:
    """Creates the networkx snn for a run configuration for the MDSA
    algorithm."""
    if not isinstance(
        stage_1_nx_graphs["input_graph"].graph["alg_props"], dict
    ):
        raise Exception("Error, algorithm properties not set.")

    return create_MDSA_neurons(stage_1_nx_graphs["input_graph"])


@typechecked
def create_MDSA_neurons(input_graph: nx.Graph) -> nx.DiGraph:
    """Creates the neurons for the MDSA algorithm."""
    mdsa_snn = nx.DiGraph()
    # TODO get spacing from somewhere.
    spacing = 0.25
    connecting_nodes = create_connecting_node(spacing)
    mdsa_snn.add_node(connecting_nodes)
    for node_index in input_graph.nodes:
        spike_once_node = create_spike_once_node(spacing, node_index)
        mdsa_snn.add_node(spike_once_node)
    return mdsa_snn


@typechecked
def create_connecting_node(spacing: float) -> LIF_neuron:
    """Creates the neuron settings for the connecting node in the MDSA
    algorithm."""
    return LIF_neuron(
        name="connecting_node",
        bias=0.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(float(-spacing), float(spacing)),
    )


def create_spike_once_node(spacing: float, node_index: int) -> LIF_neuron:
    """Creates the neuron settings for the spike_once node in the MDSA
    algorithm."""

    return LIF_neuron(
        name="spike_once",
        bias=2.0,
        du=0.0,
        dv=0.0,
        vth=1.0,
        pos=(float(0), float(node_index * 4 * spacing)),
        identifiers=[
            Identifier(description="node_index", position=0, value=node_index)
        ],
        recurrent_synapses=[
            Recurrent_synapse(
                weight=-10,  # TODO: explain why 10.
                delay=0,
                change_per_t=0,
            )
        ],
    )
