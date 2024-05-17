.. _linearize:

=========
Linearize
=========

Create a linearize model based on a given input model.

The linearization procedure will produce and run two new models. Given an input model,
a derivative model is created which will extract all derivatives for ETAs and EPSILONs.
This is followed by the creation of a linearized model with input values of the ETAs
updated according to the results from the derivative model.

~~~~~~~
Running
~~~~~~~

To create a linearized model, please run

.. pharmpy-code::

    from pharmpy.modeling import load_example_model
    from pharmpy.tools import run_linearize

    start_model = load_example_model("pheno")
    linres = run_linearize(start_model)
    linearized_model = linres.final_model
    
~~~~~~~~~
Arguments
~~~~~~~~~

+-----------------------------------------------+--------------------------------------------------------------------+
| Argument                                      | Description                                                        |
+===============================================+====================================================================+
| ``model``                                     | Input model to linaerize                                           |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``model_name``                                | Name to use for the linearized model. Default is "linbase"         |
+-----------------------------------------------+--------------------------------------------------------------------+
| ``description``                               | New description for linaerized model. Default is ""                |
+-----------------------------------------------+--------------------------------------------------------------------+
    

~~~~~~~~~~~~~~~~
De-linearization
~~~~~~~~~~~~~~~~

A linearized model can be de-linearized as well. For this, a base model is required
for parameters not stored in the linearized model. Such as thetas.

.. pharmpy-code::

    from pharmpy.tools import delinearize_model
    
    start_model = read_model('path/to/model')
    res = run_linearize(start_model)
    linearized_model = res.final_model
    
    delinearized_model = delinearize_model(linearized_model, start_model)
    
For this tool, the option ``param_mapping`` can also be set, if the ETAs from the linearized
model should be mapped to some other parameter in the base model. An example of this can be
seen below.

.. note::
    If ``param_mapping`` is used, then all ETAs in the linearized model are expected to
    be mapped to a parameter.

.. pharmpy-code::

    from pharmpy.modeling import load_example_model
    from pharmpy.tools import delinearize_model
    
    start_model = load_example_model("pheno")
    linres = run_linearize(start_model)
    linearized_model = linres.final_model
    
    param_mapping = {"ETA_1":"V", "ETA_2": "CL"}
    
    delinearized_model = delinearize_model(linearized_model, start_model)

~~~~~~~~~~~~~~~~~~~~~
The linearize results
~~~~~~~~~~~~~~~~~~~~~

OFVs
~~~~

The OFVs of the input model and the linearized model before and after estimation are summarized in the ``ofv`` table. These values should be close. A difference signals problems with the linearization.

.. pharmpy-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    res = read_results('tests/testdata/results/linearize_results.json')
    res.ofv


Individual OFVs
~~~~~~~~~~~~~~~

The individual OFVs for the base and linearized models together with their difference is in the ``iofv`` table. If there was a deviation in the ``ofv`` these values can be used to see if some particular individual was problematic to linearize.

.. pharmpy-execute::
    :hide-code:

    res.iofv

This is also plotted in ``iofv_plot``

.. pharmpy-execute::
    :hide-code:

    res.iofv_plot
