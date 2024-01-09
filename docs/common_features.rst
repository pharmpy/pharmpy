.. _common_features:

~~~~~~~~~~~~~~~
Common features
~~~~~~~~~~~~~~~

.. note::

    The AMD tool is a collection of tools, meaning that some aspects of this section may not be applicable. For
    example, datasets will be in the database for each subtool.
.. _tool_database:

Tool database
~~~~~~~~~~~~~

When running a tool, all candidate models and their corresponding files are stored in a tool database. As an example,
using the Modelsearch tool with NONMEM for estimation, a tool database might look like this:

.. code-block::

    modelsearch_dir1/
    ├── metadata.json
    ├── models
    │   ├── .datasets
    │   │   ├── .hash
    │   │   │   └── ...
    │   │   ├── modelsearch_candidate1.csv
    │   │   ├── modelsearch_candidate1.datainfo
    │   │   ├── modelsearch_candidate2.csv
    │   │   └── modelsearch_candidate2.datainfo
    │   ├── modelsearch_candidate1
    │   │   ├── modelsearch_candidate1.ext
    │   │   ├── modelsearch_candidate1.lst
    │   │   ├── modelsearch_candidate1.mod
    │   │   ├── modelsearch_candidate1.phi
    │   │   ├── mytab_mox1
    │   │   ├── nonmem.json
    │   │   ├── .pharmpy
    │   │   │   └── ...
    │   │   ├── stderr
    │   │   └── stdout
    │   ├── modelsearch_candidate2
    │   │   └── ...
    │   ├── modelsearch_candidate3
    │   │   └── ...
    │   └── modelsearch_candidate4
    │       └── ...
    ├── results.csv
    └── results.json

The top level ``modelsearch_dir1/`` will contain files relevant for the whole tool, such as metadata and results.
The subdirectory ``models/`` will contain subfolders for each model candidate, as well as a directory for all unique
datasets from the tool (see :ref:`tool_datasets`). Each candidate directory will contain the resulting NONMEM files, as
well as NONMEM output in ``stderr`` and ``stdout``.

Running directory
=================

.. note::
    This information is for NONMEM runs as it is the only estimation software currently fully supported for running
    tools.

All tools are run in a temporary directory. This means, when Pharmpy starts a tool, the model candidates and their
datasets will be written to the temporary directory, and all estimations will be started there. When model has been
estimated, it will be copied to the tool database. The dataset will also be copied to ``.datasets/`` `if that dataset
has not been stored in the database yet`. Which dataset that is copied is dependent on which of the candidate models
that have finished running first. See :ref:`tool_datasets` for more information.

Results
~~~~~~~

In general, the result objects will consist of a collection of summary tables. These can be accessed directly in your
script:

.. pharmpy-code::

    res = run_modelsearch('ABSORPTION(ZO);PERIPHERALS(1)',
                          'reduced_stepwise',
                          model=start_model,
                          results=start_model_results,
                          iiv_strategy='absorption_delay',
                          rank_type='bic',
                          cutoff=None)
    res.summary_models

For a more detailed description of which results are available, please check the documentation for each tool.

Additionally, Pharmpy will create at least two files: ``results.csv`` and ``results.json``. The .csv is intended as
a way to quickly look over your results, while the .json is a way to recreate the results object. This allows for
access to the different summary tables as data frames, and is intended to use to programmatically interact with the
results.

.. pharmpy-code::

    res = read_results('path/to/results.json')
    res.summary_models

It is also possible to read in models from the :ref:`tool_database` via the :py:func:`pharmpy.tools.retrieve_models`
function.

Metadata
~~~~~~~~

For each tool run, a metadata file ``metadata.json`` will be created. This contains information about start- and end
time, which options were used, which Pharmpy version etc. Example of metadata-file:

.. pharmpy-execute::
    :hide-code:

    with open('tests/testdata/results/metadata.json') as f:
        print(f.read())


.. _tool_datasets:

Datasets
~~~~~~~~

Pharmpy will create a directory ``.datasets/`` where any unique datasets the tool creates will be stored. An example
of this is when running Modelsearch and having zero order absorption in the search space, a RATE column will be
created. If any of the stepwise algorithms are used, the subsequent models will have the "same" dataset, and thus only
one copy of that dataset will be located in ``.datasets/``.
