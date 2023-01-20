"""Tests whether the snn MDSA algorithm results with adaptation equal those of
the default/Neumann implementation."""
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
        """Tests whether the MDSA algorithm with adaptation yields the same
        results as without adaptation."""
        for redundancy in range(2, 6, 2):
            # Modify configuration to include adaptation.
            self.mdsa_settings.adaptations = {"redundancy": [redundancy]}
            self.mdsa_settings.overwrite_images_only = False
            self.mdsa_settings.export_images = False

            # Narrow down test scope by overriding experiment settings.
            # self.mdsa_settings.size_and_max_graphs = [(4, 1)]
            self.mdsa_settings.algorithms = {
                "MDSA": [
                    {"m_val": 0},
                    {"m_val": 1},
                ]
            }

            # Perform test.
            self.helper(self.mdsa_settings)
