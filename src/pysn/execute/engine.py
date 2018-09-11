# -*- encoding: utf-8 -*-
"""
================
Execution Engine
================

Job *creator* for a :class:`~pysn.generic.Model` implementation.

The critical, non-agnostic and central unit to inherit (e.g.
:class:`~pysn.api_nonmem.execute.NONMEM7`). *That implementation* can be multiclassed dynamically
for :class:`~pysn.tool.Tool` implementations, if mere duck typing doesn't cut it.

.. note:: An :class:`.Engine` implementation is expected to implement all methods/attributes.

Definitions
-----------
"""

from .environment import SystemEnvironment


class Engine:
    """An execution engine (e.g. for NONMEM7).

    Arguments:
        envir: Type of :class:`~.environment.Environment` to use.

    Default `envir` is a subclass (depending on the current OS) of
    :class:`~.environment.SystemEnvironment` (direct execution). An :class:`~.self` object evaluates
    True *if and only if* an estimation task can be initiated *now*.
    """

    def __init__(self, envir=None):
        if envir:
            self.environment = envir
        else:
            self.environment = SystemEnvironment()

    def estimate(self, models, **options):
        """Starts estimation of one/many models and returns :class:`~.job.Job` object.

        Arguments:
            models: List of :class:`~pysn.generic.Model` objects to estimate.
            **options: Estimation options to pass on to :class:`~.engine.Engine`."""
        commands = [self._estimate_command(model) for model in models]
        job = self.environment.submit(*commands)
        return job

    @property
    def bin(self):
        """Path to main binary."""
        raise NotImplementedError

    @property
    def version(self):
        """Version (of main binary)."""
        raise NotImplementedError

    def _estimate_command(self, model):
        """Creates the command line to estimate 'model'."""
        raise NotImplementedError

    def __bool__(self):
        """Should only eval True if engine is ready for immediate execution, in the present."""
        return False
