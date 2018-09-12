# -*- encoding: utf-8 -*-
"""
================
Execution Engine
================

Job creator for a :class:`~pysn.generic.Model` implementation.

The critical, non-agnostic and central unit to inherit (e.g.
:class:`~pysn.api_nonmem.execute.NONMEM7`). That implementation can be multiclassed dynamically
for :class:`~pysn.tool.Tool` implementations, if mere duck typing doesn't cut it.

.. note:: An :class:`.Engine` implementation is expected to implement all methods/attributes.

Definitions
-----------
"""

from .environment import SystemEnvironment
from .run_directory import RunDirectory


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

    def evaluate(self, model, cwd=None, **kwds):
        """Starts evaluation of a model and returns :class:`~.job.Job` object.

        Arguments:
            model: The :class:`~pysn.generic.Model` object to evaluate.
            cwd: Directory to create run directory in (temporary if None).
            **kwds: Extra evaluation options.

        .. todo:: Make either this level, or one up (:class:`pysn.generic.Model` and
            :class:`pysn.tool.Tool`), emulate -model_subdir command of PsN. Estimating
            pheno_real.mod with no dir info but the *parent* (otherwise it must be a temp dir),
            should create 'pheno_real/' dir only if it doesn't already exist, and *then* create
            pheno_real/estimate_dir1/, pheno_real/evaluate_dir74/, etc. (be clear but not collide).
        """

        rundir = RunDirectory(cwd, model.path.stem)
        command = self.get_commandline('evaluate', model)
        return self.environment.submit(command, rundir)

    def estimate(self, model, cwd=None, **kwds):
        """Starts estimation of a model and returns :class:`~.job.Job` object.

        Arguments:
            model: The :class:`~pysn.generic.Model` object to estimate.
            cwd: Directory to create run directory in (temporary if None).
            **kwds: Extra estimation options.
        """

        rundir = RunDirectory(cwd, model.path.stem)
        command = self.get_commandline('estimate', model)
        return self.environment.submit(command, rundir)

    @property
    def bin(self):
        """Path (to main binary)."""
        raise NotImplementedError

    @property
    def version(self):
        """Version (of main binary)."""
        raise NotImplementedError

    def get_commandline(self, task, model):
        """Returns a command line for performing a task on a model.

        Arguments:
            task: Any task of {'evaluate', 'estimate'}.
            model: A :class:`pysn.generic.Model` object to perform 'task' on.
        """
        raise NotImplementedError

    def __bool__(self):
        """True if this engine is ready for executing (at any time)."""
        return False
