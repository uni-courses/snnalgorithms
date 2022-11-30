"""Creates the MDSA snn."""


import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked


@typechecked
def get_new_mdsa_graph(stage_1_nx_graphs: dict) -> nx.DiGraph:
    """Creates the networkx snn for a run configuration for the MDSA
    algorithm."""
    if not isinstance(
        stage_1_nx_graphs["input_graph"].graph["alg_props"], dict
    ):
        raise Exception("Error, algorithm properties not set.")

    return create_MDSA_neurons()


@typechecked
def create_MDSA_neurons() -> nx.DiGraph:
    """Creates the neurons for the MDSA algorithm."""
    mdsa_snn = nx.DiGraph()
    # TODO get spacing from somewhere.
    connecting_nodes = create_connecting_node(0.25)
    mdsa_snn.add_node(connecting_nodes)
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
