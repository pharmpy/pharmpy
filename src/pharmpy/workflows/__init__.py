"""
=========
workflows
=========


Central support for generation of workflows and executing models for evaluation, estimation
and simulation tasks. The workflows themselves are implemented in the different methods modules.

This package defines the following classes:

    .. list-table::
        :widths: 20 80
        :stub-columns: 1

        - - :class:`~pharmpy.workflows.RunDirectory`
          - Run directory with cleaning and safe unlink operations.
            Invoking directory, where are the models? Which files to copy where?
        - - :class:`~pharmpy.workflows.ExecutionEnvironment`
          - Platform (e.g. Linux) & system (e.g. SLURM) implementation.
            The cluster/local or OS etc. to run jobs on.
        - - :class:`~pharmpy.workflows.ModelExecutionEngine`
          - The Engine to use for performing a modelling task, e.g. NONMEM or nlmixr
        - - :class:`~pharmpy.workflows.WorkflowEngine`
          - The Engine to use to run and orchestrate a workflow
        - - :class:`~pharmpy.workflows.Workflow`
          - An abstract representation of an actual workflow

Where :class:`~pharmpy.workflow.Engine` is the **critical unit** for an implementation to
inherit.

:class:`~pharmpy.execute.environment.Environment` require inheritance for a specific purpose (e.g.
Linux, SLURM, dummy directory, etc.).
:class:`~pharmpy.execute.run_directory.RunDirectory` and :class:`~pharmpy.execute.job.Job` will
likely remain model (type) agnostic.

Definitions
-----------
"""

from .engine import ModelExecutionEngine
from .environment import ExecutionEnvironment
from .run_directory import RunDirectory
from .workflow import Workflow
from .workflow_engine import WorkflowEngine

__all__ = [
    'ModelExecutionEngine',
    'ExecutionEnvironment',
    'RunDirectory',
    'WorkflowEngine',
    'Workflow',
]
