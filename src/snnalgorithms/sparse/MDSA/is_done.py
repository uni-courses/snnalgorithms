"""Determines whether the snn algorithm is done."""
from typing import Dict

import networkx as nx
from typeguard import typechecked


@typechecked
def mdsa_is_done(run_config: Dict, snn_graph: nx.DiGraph, t: int) -> bool:
    """Checks whether the network is done or not.

    First checks if terminator node is done. Then checks, if that is not
    the case whether radiation is active.
    """
    if list(run_config["algorithm"].keys()) == ["MDSA"]:
        if snn_graph.nodes["terminator_node"]["nx_lif"][t].spikes:
            print("terminated")
            return True
        if run_config["radiation"] is not None:
            # Radiation may have killed any neuron. This may have arbitrarily
            # caused the neuron to not spike. This algorithm requires that
            # at least 1 selector neuron is firing within if t>1.
            for node_name in snn_graph.nodes:
                if "selector" in node_name:
                    if (
                        snn_graph.nodes[node_name]["nx_lif"][t].spikes
                        and t > 0
                    ):
                        print(f"t={t},false1, nodename={node_name}")
                        return False
                elif "terminator" in node_name:
                    if (
                        snn_graph.nodes[node_name]["nx_lif"][t].spikes
                        and t > 0
                    ):
                        return True
            if t > 0:
                print(f"t={t}, done.")
                return True
            print(f"t={t},false2")
            return False
        print(f"t={t},false3")
        return False
    raise Exception("Algorithm termination mode not yet found.")
