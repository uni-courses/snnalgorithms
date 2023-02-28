"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.

from snnalgorithms.sparse.Discovery.Discovery import Discovery, Discovery_algo

# mdsa = MDSA(list(range(0, 4, 1)))
# mdsa_configs = get_algo_configs(algo_spec=mdsa.__dict__)
# verify_algo_configs(algo_name="MDSA", algo_configs=mdsa_configs)

# Parse command line interface arguments to determine what this script does.
# args = parse_cli_args()
disco = Discovery()
Discovery_algo(disco=disco)
