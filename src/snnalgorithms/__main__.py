"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.

from src.snnalgorithms.get_alg_configs import (
    get_algo_configs,
    verify_algo_configs,
)
from src.snnalgorithms.population.MDSA import MDSA

mdsa = MDSA(list(range(0, 4, 1)))
mdsa_configs = get_algo_configs(mdsa.__dict__)
verify_algo_configs("MDSA", mdsa_configs)

# Parse command line interface arguments to determine what this script does.
# args = parse_cli_args()
