.. _options:

~~~~~~~
Options
~~~~~~~

Running a tool requires the user to specify the input to the tool and in many cases to give options that can change the beahviour of the run.
These options are either specific to the particular tool or general options that can be given to any tool. There are three categories of options
a tool can take:


+-----------------------------+-------------------------------------------------------------------------------------------------+
| Type                        | Description                                                                                     |
+=============================+=================================================================================================+
| :code:`tool_options`        | Options specific to one tool                                                                    |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`common_options`      | Options common to all tools                                                                     |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`dispatching_options` | Options affecting how a tool is dispatched. These shouldn't directly affect what the tool does. |
+-----------------------------+-------------------------------------------------------------------------------------------------+

The values of all types of options can be found in the metadata-file for each tool and subtool run.

Tool options
~~~~~~~~~~~~

Since the tool options are specific for each tool their documentation can be found under the user guides for the separate tools.

Common options
~~~~~~~~~~~~~~

Common options are options that can directly affect the execution of a tool and not only the environment in which the tool is run (see the dispatching options). 

+-----------------------------+-------------------------------------------------------------------------------------------------+
| Option                      | Description                                                                                     |
+=============================+=================================================================================================+
| :code:`esttool`             | The external tool to use for model estimation and evaluation.                                   |
|                             | The options are :code:`nonmem`, :code:`nlmixr` and :code:`dummy`. The dummy estimation tool     |
|                             | doesn't do estimation, but instead randomizes the results. Since this is very fast it can be    |
|                             | used for testing and demonstration purposes.                                                    |
|                             | :code:`nonmem` is the default                                                                   |
+-----------------------------+-------------------------------------------------------------------------------------------------+

Dipatching options
~~~~~~~~~~~~~~~~~~

Dipatching options are technical options telling Pharmpy something about how a workflow should be dispatched, i.e. run. They are not intended to be able to directly change any results of a tool just how they are executed.

.. warning::

    In practice results might still change when changing dispatching options due to unforeseen details of the underlying system.

+-----------------------------+-------------------------------------------------------------------------------------------------+
| Option                      | Description                                                                                     |
+=============================+=================================================================================================+
| :code:`broadcaster`         | The broadcaster to use for broadcasting log messages during the run. The options are            |
|                             | :code:`terminal` and :code:`null`. :code:`terminal` is the default and outputs messages to the  |
|                             | terminal. :code:`null` will turn off broadcasting.                                              |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`dispatcher`          | The dispatcher use for executing the workflow. The options are :code:`local_dask` and           |
|                             | :code:`local_serial`. Default is :code:`local_dask`. For more information see                   |
|                             | :ref:`dispatching`                                                                              |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`name`                | The name of the tool run. For more information see :ref:`context`                               |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`ncores`              | Maximum number of CPU cores to use. Default is to use all available cores on the computer.      |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`ref`                 | The path to the tool run. For more information see :ref:`context`                               |
+-----------------------------+-------------------------------------------------------------------------------------------------+

