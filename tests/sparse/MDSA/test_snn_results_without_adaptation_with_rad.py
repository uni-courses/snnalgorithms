"""Tests whether the snn MDSA algorithm results equal those of the
default/Neumann implementation."""
# pylint: disable=R0801

from typeguard import typechecked

from tests.sparse.MDSA.test_snn_results import Test_mdsa_snn_results


class Test_mdsa_snn_results_with_adaptation(Test_mdsa_snn_results):
    """Tests whether the snn implementation of the MDSA algorithm with
    adaptation yields the same results as the default/Neumann implementation if
    its weights are identical."""

    # Initialize test object
    #
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super(Test_mdsa_snn_results, self).__init__(*args, **kwargs)
        # Generate default experiment config.
        self.create_exp_config()

    @typechecked
    def test_something(self) -> None:
        """Tests whether the SNN MDSA algorithm without adaptation yields the
        same results as the original Neumann version of the MDSA algorithm."""

        self.mdsa_settings.overwrite_images_only = True
        self.mdsa_settings.export_images = True
        self.mdsa_settings.radiations = {"neuron_death": [0.25]}

        # Narrow down test scope by overriding experiment settings.
        # self.mdsa_settings.size_and_max_graphs = [(4, 1)]
        self.mdsa_settings.algorithms = {
            "MDSA": [
                # {"m_val": 0},
                {"m_val": 1},
            ]
        }

        self.helper(self.mdsa_settings)
