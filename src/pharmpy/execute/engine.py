# -*- encoding: utf-8 -*-
"""
================
Execution Engine
================

Job creator for a :class:`~pharmpy.generic.Model` implementation.

The critical, non-agnostic and central unit to inherit (e.g.
:class:`~pharmpy.api_nonmem.execute.NONMEM7`). That implementation can be multiclassed dynamically
for :class:`~pharmpy.tool.Tool` implementations, if mere duck typing doesn't cut it.

.. note:: An :class:`.Engine` implementation is expected to implement all methods/attributes.

Definitions
-----------
"""

from .environment import SystemEnvironment
from .run_directory import RunDirectory


class Engine:
    """An execution engine (e.g. for NONMEM7) of a :class:`~pharmpy.generic.Model`.

    Is a model API attached to attribute :attr:`Model.execute <pharmpy.generic.Model.execute>`.

    Arguments:
        envir: Type of :class:`~.environment.Environment` to use.

    Default `envir` inherits (depending on the current OS) :class:`~.environment.SystemEnvironment`
    (direct execution).
    """

    def __init__(self, model, envir=None):
        self.model = model
        if envir:
            self.environment = envir
        else:
            self.environment = SystemEnvironment()

    async def evaluate(self, cwd=None):
        """Starts model evaluation and returns :class:`~.job.Job` object.

        Arguments:
            cwd: Directory to create run directory in (temporary if None).

        .. todo:: Make either this level, or one up (:class:`pharmpy.generic.Model` and
            :class:`pharmpy.tool.Tool`), emulate -model_subdir command of PsN. Estimating
            pheno_real.mod with no dir info but the *parent* (otherwise it must be a temp dir),
            should create 'pheno_real/' dir only if it doesn't already exist, and *then* create
            pheno_real/estimate_dir1/, pheno_real/evaluate_dir74/, etc. (be clear but not collide).
        """

        raise NotImplementedError()

    async def estimate(self, cwd=None):
        """Starts model estimation and returns :class:`~.job.Job` object.

        Arguments:
            cwd: Directory to create run directory in (temporary if None).
        """

        return await self._execute_model(cwd)

    @property
    def bin(self):
        """Path (to main binary)."""
        raise NotImplementedError

    @property
    def version(self):
        """Version (of main binary)."""
        raise NotImplementedError

    def get_commandline(self, task):
        """Returns a command line for performing a task.

        Arguments:
            task: Any task of {'evaluate', 'estimate'}.
        """
        raise NotImplementedError

    async def _execute_model(self, cwd):
        rundir = RunDirectory(cwd, self.model.path.stem)
        rundir.model = self.model
        command = rundir.model.execute.get_commandline('estimate')
        envir = rundir.model.execute.environment

        job = await envir.submit(command, rundir)
        rundir.bind_job(job)
        return rundir

    def __bool__(self):
        """True if this engine is ready for executing *now*."""
        return False
