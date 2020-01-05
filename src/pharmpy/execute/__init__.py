"""
===============
Model Execution
===============

Central support for executing models for evaluation, estimation and simulation tasks. Useful for:

    - :class:`~pharmpy.model.Model` objects to, for example, evaluate themselves on request
    - Full workflows in implementations of :class:`~pharmpy.tool.Tool` (e.g. FREM)

This package defines four classes:

    .. list-table::
        :widths: 20 80
        :stub-columns: 1

        - - :class:`~pharmpy.execute.job.Job`
          - A job unit.
            Contains an asynchronous process and exposes blocking and non-blocking stdout.
        - - :class:`~pharmpy.execute.run_directory.RunDirectory`
          - Run directory with cleaning and safe unlink operations.
            Invoking directory, where are the models? Which files to copy where?
        - - :class:`~pharmpy.execute.environment.Environment`
          - Platform (e.g. Linux) & system (e.g. SLURM) implementation.
            The cluster/local or OS etc. to start jobs on.
        - - :class:`~pharmpy.execute.engine.Engine`
          - Creates Job objects, executing some task, for a *specific* implementation.
            Contains Environment object. The focal point for implementation inheritance.

Where :class:`~pharmpy.execute.engine.Engine` is the **critical unit** for a implementation to
inherit, e.g. :class:`~pharmpy.api_nonmem.execute.NONMEM7`.

:class:`~pharmpy.execute.environment.Environment` require inheritance for a specific purpose (e.g.
Linux, SLURM, dummy directory, etc.), but not an implementation.
:class:`~pharmpy.execute.run_directory.RunDirectory` and :class:`~pharmpy.execute.job.Job` will
likely remain model (type) agnostic.

Definitions
-----------
"""

from .engine import Engine
from .environment import Environment

__all__ = ['Engine', 'Environment']
