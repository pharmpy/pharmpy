.. _vpc:

===
VPC
===

Create a visual predicitive check plot.

~~~~~~~
Running
~~~~~~~

The VPC tool is available both in Pharmpy/pharmr and from the command line.

.. pharmpy-code::

    from pharmpy.modeling import read_model
    from pharmpy.tools import read_modelfit_results, run_vpc

    model = read_model('path/to/model')
    model_results = read_modelfit_results('path/to/model')
    res = run_vpc(model=model, results=model_results, samples=300)

Example of running VPC from the command line:

.. code::

    pharmpy run vpc run1.mod --samples=300

~~~~~~~~~
Arguments
~~~~~~~~~

Mandatory
---------

+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| Argument                                          | Description                                                                             |
+===================================================+=========================================================================================+
| ``model``                                         | Start model                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``results``                                       | ModelfitResults for the start model                                                     |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``samples``                                       | Number of simulation samples. Minimum 20                                                |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+
| ``stratify``                                      | Name of column to stratify on.                                                          |
+---------------------------------------------------+-----------------------------------------------------------------------------------------+


~~~~~~~~~~~~~~~
The VPC results
~~~~~~~~~~~~~~~

The results object contains the VPC plot.


.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/vpc_results.json')
    res.plot
