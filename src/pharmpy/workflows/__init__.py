"""
.. list-table:: Options for the data module
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``default_dispatcher``
     - ``pharmpy.workflows.local_dask``
     - str
     - Name of default dispatcher module
   * - ``default_model_database``
     - ``pharmpy.workflows.LocalDirectoryDatabase``
     - str
     - Name of default model database class
   * - ``default_tool_database``
     - ``pharmpy.workflows.LocalDirectoryToolDatabase``
     - str
     - Name of default tool database class

"""

import importlib

import pharmpy.config as config

from .args import split_common_options
from .call import call_workflow
from .dispatchers import local_dask
from .execute import execute_workflow
from .log import Log
from .model_database import (
    LocalDirectoryDatabase,
    LocalModelDirectoryDatabase,
    ModelDatabase,
    NullModelDatabase,
)
from .task import Task
from .tool_database import LocalDirectoryToolDatabase, NullToolDatabase, ToolDatabase
from .workflow import Workflow


class WorkflowConfiguration(config.Configuration):
    module = 'pharmpy.workflows'
    default_dispatcher = config.ConfigItem(
        'pharmpy.workflows.local_dask', 'Name of default dispatcher module'
    )
    default_model_database = config.ConfigItem(
        'pharmpy.workflows.LocalDirectoryDatabase', 'Name of default model database class'
    )
    default_tool_database = config.ConfigItem(
        'pharmpy.workflows.LocalDirectoryToolDatabase', 'Name of default tool database class'
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


default_dispatcher = _importclass(conf.default_dispatcher)
default_model_database = _importclass(conf.default_model_database)
default_tool_database = _importclass(conf.default_tool_database)


__all__ = [
    'call_workflow',
    'default_dispatcher',
    'default_model_database',
    'default_tool_database',
    'execute_workflow',
    'split_common_options',
    'local_dask',
    'LocalDirectoryDatabase',
    'LocalModelDirectoryDatabase',
    'LocalDirectoryToolDatabase',
    'Log',
    'NullModelDatabase',
    'NullToolDatabase',
    'ModelDatabase',
    'Task',
    'ToolDatabase',
    'Workflow',
]
