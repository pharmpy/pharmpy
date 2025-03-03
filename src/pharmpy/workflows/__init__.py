"""
.. list-table:: Options for the data module
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``default_model_database``
     - ``pharmpy.workflows.LocalDirectoryDatabase``
     - str
     - Name of default model database class
   * - ``default_context``
     - ``pharmpy.workflows.LocalDirectoryContext``
     - str
     - Name of default context class

"""

import importlib

import pharmpy.config as config

from .args import split_common_options
from .contexts import Context, LocalDirectoryContext
from .execute import execute_subtool, execute_workflow
from .log import Log
from .model_database import (
    LocalDirectoryDatabase,
    LocalModelDirectoryDatabase,
    ModelDatabase,
    NullModelDatabase,
)
from .model_entry import ModelEntry
from .results import ModelfitResults, Results, SimulationResults
from .task import Task
from .workflow import Workflow, WorkflowBuilder


class WorkflowConfiguration(config.Configuration):
    module = 'pharmpy.workflows'
    default_model_database = config.ConfigItem(
        'pharmpy.workflows.LocalDirectoryDatabase', 'Name of default model database class'
    )
    default_context = config.ConfigItem(
        'pharmpy.workflows.LocalDirectoryContext', 'Name of default context class'
    )


conf = WorkflowConfiguration()


def _importclass(name):
    a = name.split('.')
    module_name = '.'.join(a[:-1])
    if module_name == __name__:
        return globals()[a[-1]]
    else:
        module = importlib.import_module(module_name)
        return module.a[-1]


default_model_database = _importclass(conf.default_model_database)
default_context = _importclass(conf.default_context)


class DispatchingError(Exception):
    pass


__all__ = [
    'default_model_database',
    'default_context',
    'execute_subtool',
    'execute_workflow',
    'split_common_options',
    'DispatchingError',
    'LocalDirectoryDatabase',
    'LocalModelDirectoryDatabase',
    'LocalDirectoryContext',
    'Log',
    'NullModelDatabase',
    'ModelDatabase',
    'ModelEntry',
    'ModelfitResults',
    'Results',
    'SimulationResults',
    'Task',
    'Context',
    'Workflow',
    'WorkflowBuilder',
]
