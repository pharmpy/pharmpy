.. _simulation:


===========
Simulation
===========

The simulation tool is a tool for running simulations.

~~~~~~~
Running
~~~~~~~

The simulation tool is available both in Pharmpy/pharmr and from the command line.

To initiate the simulation tool in Python/R:

.. pharmpy-code::

    from pharmpy.tools import run_simulation

    model = read_model('path/to/model')
    res = run_simulation(model)


In order to run a simulation for a model the model needs to have DV in the output table and a simulation step.
A simulation step can be added using ``set_simulation(model, n=N)`` with ``N`` being the number of simulations.


Arguments
~~~~~~~~~

The input argument to the simulation tool is a model object containing a simulation step.

+-------------------------------------------------+------------------------------------------------------------------+
| Argument                                        | Description                                                      |
+=================================================+==================================================================+
| ``model``                                       | Model                                                            |
+-------------------------------------------------+------------------------------------------------------------------+

~~~~~~~~
Examples
~~~~~~~~

Example of adding a simulation step to a model and running the simulation tool:

.. pharmpy-code::

    from pharmpy.modeling import set_simulation
    from pharmpy.tools import run_simulation

    model = read_model('path/to/model')
    model = set_simulation(model, n=300)
    res = run_simulation(model)


~~~~~~~~~~~~~~~~~~~~~~
The Simulation results
~~~~~~~~~~~~~~~~~~~~~~

The results of the simulation will be stored in the results.csv file. The simulation ``table`` contains the simulation
number, index and DV value:

.. pharmpy-code::

    from pharmpy.tools import run_simulation

    model = read_model('path/to/model')
    res = run_simulation(model)
    res.table

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/simulation_results.json')
    res.table
