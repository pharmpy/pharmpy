# -*- encoding: utf-8 -*-
"""
===============
Model Execution
===============

Central support for executing models for evaluation, estimation and simulation tasks. Useful for:

    - :class:`~pysn.model.Model` objects to, for example, evaluate themselves on request
    - Full workflows in implementations of :class:`~pysn.tool.Tool` (e.g. FREM)

The package defines four classes:

    .. list-table::
        :widths: 20 80
        :stub-columns: 1

        - - :class:`~pysn.execute.job.Job`
          - A job unit. Can contain multiple processes.

        - - :class:`~pysn.execute.run_directory.RunDirectory`
          - Run directory with cleaning and safe unlink operations.

        - - :class:`~pysn.execute.environment.Environment`
          - Platform (e.g. Linux) & system (e.g. SLURM) implementation.

        - - :class:`~pysn.execute.engine.Engine`
          - Job creator for some purpose. Contains Environment object.

Where :class:`~pysn.execute.engine.Engine` is the **critical unit** for a implementation to inherit,
e.g. :class:`~pysn.api_nonmem.execute.NONMEM7`. Job, RunDirectory and Environment do require
inheritance for specific purposes (e.g. Linux, SLURM, dummy directory, etc.) but they remain,
ideally, model (type) agnostic and "implementations" can thus be pushed upstream.

Definitions
-----------
"""

from .engine import Engine
from .environment import Environment

__all__ = ['Engine', 'Environment']
