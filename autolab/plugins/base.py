"""Plugin interface for autolab extensions."""


class AutolabPlugin:
    """Base class for autolab plugins.

    Override any hook method to receive callbacks during the experiment lifecycle.
    """

    name = "base"

    def on_experiment_start(self, exp_num, config):
        """Called when an experiment begins."""

    def on_experiment_end(self, exp_num, results):
        """Called when an experiment completes."""

    def on_ralph_iteration(self, iteration, strategy, gap, result):
        """Called after each ralph-loop iteration."""

    def on_knowledge_update(self, section, entry):
        """Called when REGISTRY.md is updated."""
