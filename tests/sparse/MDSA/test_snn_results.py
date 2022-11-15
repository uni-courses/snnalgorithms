"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801
import os
import shutil
import unittest

from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    experiment_config_for_mdsa_testing,
)
from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.Experiment_runner import Experiment_runner
from typeguard import typechecked

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA


class Test_mdsa_snn_results(unittest.TestCase):
    """Tests whether the snn implementation of the MDSA algorithm yields the
    same results as the default/Neumann implementation if its weights are
    identical."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.algorithms = {
            "MDSA": get_algo_configs(MDSA(list(range(0, 2, 1))).__dict__)
        }

    @typechecked
    def test_snn_results_equal_neumann_results(self) -> None:
        """Tests whether the results of the snn implementation of the MDSA
        algorithm are the same as those of the default/Neumann implementation
        of that MDSA algorithm. ."""

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("latex"):
            shutil.rmtree("latex")

        # Generate default experiment config.
        mdsa_creation_only_size_3_4: dict = (
            experiment_config_for_mdsa_testing()
        )

        # Do not output images.
        mdsa_creation_only_size_3_4["overwrite_visualisation"] = True
        mdsa_creation_only_size_3_4["show_snns"] = False
        mdsa_creation_only_size_3_4["export_images"] = False

        # Include desired mdsa settings.
        mdsa_creation_only_size_3_4["algorithms"] = self.algorithms

        verify_experiment_config(
            Supported_experiment_settings(),
            mdsa_creation_only_size_3_4,
            has_unique_id=False,
            allow_optional=True,
        )

        # Do not apply adaptation (default).
        # Do not apply radiation (default).

        # Verify results are identical.
        Experiment_runner(mdsa_creation_only_size_3_4)
