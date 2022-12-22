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
        self.create_exp_setts()

    @typechecked
    def test_something(self) -> None:
        """Tests whether the MDSA algorithm with adaptation yields the same
        results as without adaptation."""
        for redundancy in range(1, 5):
            # Modify configuration to include adaptation.
            self.mdsa_settings["adaptations"] = {"redundancy": [redundancy]}
            self.helper(self.mdsa_settings)
