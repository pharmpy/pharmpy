# -*- encoding: utf-8 -*-
"""
================
Execution Engine
================

Job *creator* for a :class:`~pysn.model.Model` implementation.

The critical, non-agnostic and central unit to inherit (e.g.
:class:`~pysn.api_nonmem.execute.NONMEM7`). *That implementation* can be multiclassed dynamically
for :class:`~pysn.tool.Tool` implementations, if mere duck typing doesn't cut it.

Definitions
-----------
"""

from .environment import SystemEnvironment


class Engine:
    """An execution engine (e.g. NONMEM or similar)."""

    def __init__(self, envir=None):
        if envir:
            self.environment = envir
        else:
            self.environment = SystemEnvironment()

    def estimate(self, models, **options):
        """Estimate Model's in the default Environment, sending optional options to Engine."""
        commands = [self.create_command(model) for model in models]
        job = self.environment.submit(*commands)
        return job

    def create_command(self, model):
        """Creates the command line to start execution."""
        raise NotImplementedError

    @property
    def bin(self):
        """Path to main binary."""
        raise NotImplementedError

    @property
    def version(self):
        """Version (of main binary)."""
        raise NotImplementedError

    def __bool__(self):
        """Should only eval True if engine is ready for immediate execution, in the present."""
        return False
