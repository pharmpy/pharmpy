.. _dispatching:

~~~~~~~~~~~
Dispatching
~~~~~~~~~~~

Dispatching is how a tool is run. Is it on a local system? Should it run in serial or parallel? 

~~~~~~~~~~~
Dispatchers
~~~~~~~~~~~

A dispatcher is the part of Pharmpy responsible for running a tool in a certain way on a certain platform. The dispatcher is responsible
for executing the tool, parallelization and communication with the target system. When running a tool it is possible to select
which dispatcher to use via the :code:`dispatcher` option (see :ref:`options`). Here is a list of the available dispatchers:

+-----------------------------+-------------------------------------------------------------------------------------------------+
| Dispatcher                  | Description                                                                                     |
+=============================+=================================================================================================+
| :code:`local_dask`          | This is the default dispatcher. It uses the dask Python library to parallelize within           |
|                             | a tool.                                                                                         |
+-----------------------------+-------------------------------------------------------------------------------------------------+
| :code:`local_serial`        | Runs the tool in serial except that it can parallelize each NONMEM run using MPI                |
+-----------------------------+-------------------------------------------------------------------------------------------------+

:code:`local_dask`
~~~~~~~~~~~~~~~~~~

The :code:`local_dask` dispatcher uses the dask Python library to parallelize within
a tool. It will run Python function calls in parallel and calls to external tools in parallel.
(e.g. NONMEM execution). One drawback of this dispatcher is that it is easy to have it run
with low efficiency if too many cores are selected. The cores will be idle if they cannot be
used and this can happen for example if there is a bottle neck part of the tool that cannot
be parallelized, for example if the tool switches beteween running many and few models in parallel.

:code:`local_serial`
~~~~~~~~~~~~~~~~~~~~

The :code:`local_serial` dispatcher runs all tasks in serial. This means that one model will be executed at a time. Runs can still be
parallelized by parallelizing the run of each model. This means that all cores will be used to parallelize NONMEM using MPI. If the dispatcher
detects that it is running on Slurm it will be able to query Slurm about which machines can be used for the parallelization.
