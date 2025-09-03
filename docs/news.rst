====
News
====


Pharmpy and pharmr 1.9.0 released
---------------------------------

*2025-09-03* --- Pharmpy 1.9.0 and pharmr 1.9.0 are now available. The highlights of the release are:

* Fixed incorrect condition number calculation in strictness evaluation. If you use this in any of the AMD-tools, please double check your results!
* New tool ModelRank to rank different models and dynamically assess parameter uncertainty, this tool is now used by all AMD-tools
* Added tool VPC to run visual predictive checks
* Support of additive error model in the RUVSearch tool
* Add documentation for Bootstrap tool, ModelRank tool, VPC tool, and selection criteria
* And various smaller functions and bug fixes

See :ref:`CHANGELOG<1.9.0>` for details.

Pharmpy now has a discussions page!
-----------------------------------

*2025-08-18* --- We now have a Discussions page on GitHub! At this page you can ask questions and discuss with other users and developers from the Pharmpy community.

Link `here <https://github.com/pharmpy/pharmpy/discussions>`_.

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
