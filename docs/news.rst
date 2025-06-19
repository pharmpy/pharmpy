====
News
====

Pharmpy and pharmr 1.8.0 released
---------------------------------

*2025-06-19* --- This is mostly a bugfix release. Most importantly a bug causing resuming a crashed amd run to fail. Adding the :code:`strictness` option to the bootstrap tool makes it almost as feature complete as the PsN bootstrap. 

See :ref:`CHANGELOG<1.8.0>` for details.

Pharmpy and pharmr 1.7.2 released
---------------------------------

*2025-05-21* --- Further testing of the rerun functionality revealed a multitude of issues. These should be fixed with the newly released Pharmpy 1.7.2 and pharmr 1.7.2.

See :ref:`CHANGELOG<1.7.2>` for details.

Pharmpy and pharmr 1.7.1 released
---------------------------------

*2025-05-14* --- A problematic issue with the rerun functionality surfaced and therefore we released Pharmpy 1.7.1 and pharmr 1.7.1 with a fix.

See :ref:`CHANGELOG<1.7.1>` for (slightly more) details.


Pharmpy and pharmr 1.7.0 released
---------------------------------

*2025-05-12* --- Pharmpy 1.7.0 and pharmr 1.7.0 are now available. The highlights of the release are:

* New dispatcher :code:`local_serial` that can be used with :code:`dispatcher="local_serial"` to run parallel NONMEM using MPI both locally and on Slurm.
* Support for directly retrieving results of Pharmpy tools when rerunning a user scripts
* Support for detecting if a tool run was interrupted upon attempting to rerun

See :ref:`CHANGELOG<1.7.0>` for the details.

New news page
-------------

*2025-05-12* --- Welcome to the new NEWS page. We will post news on releases and other topics that might be of interest, such as courses. 
