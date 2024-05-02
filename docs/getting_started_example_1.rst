.. _example1:

=========================
Simple estimation example
=========================

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')
   from docs.help_functions import print_model_diff

Here is a simple example of how to create a model and performing an estimation.

Create a model
~~~~~~~~~~~~~~

We start with creating a model.
A model can either be imported using ``read_model``:

.. pharmpy-code::

    from pharmpy.modeling import read_model

    model = read_model('path/to/model')

or it can be created from a dataset with ``create_basic_pk_model``:

.. pharmpy-code::

    from pharmpy.modeling import create_basic_pk_model

    model = create_basic_pk_model('path/to/dataset')

Lets create a basic IV PK model:

.. pharmpy-execute::
    :hide-output:

    from pharmpy.modeling import *

    dataset_path = 'tests/testdata/nonmem/pheno.dta'
    model = create_basic_pk_model(administration='iv',
                                        dataset_path=dataset_path,
                                        cl_init=0.01,
                                        vc_init=1.0,
                                        mat_init=0.1)


We can now examine the model code:

.. pharmpy-execute::
    model.statements

When creating a model with ``create_basic_pk_model`` the model will be a pharmpy model. To convert it to a NONMEM
model use:

.. pharmpy-execute::
   :hide-output:

    model = convert_model(model, 'nonmem')

We can then examine the NONMEM model code:

.. pharmpy-execute::
    print_model_code(model)


Modify model
~~~~~~~~~~~~

Now the model can be manipulated. We can for example add a peripheral compartment:

.. pharmpy-execute::
    model = add_peripheral_compartment(model)
    model.statements

We can now see that a compartment has been added to the model.

We can also remove IIV from a model parameter:

.. pharmpy-execute::
    model = remove_iiv(model, "CL")
    model.statements.find_assignment("CL")

or add IIV to a parameter:

.. pharmpy-execute::
    model = add_iiv(model, "CL", "exp")
    model.statements.find_assignment("CL")

For more information about what transformations can be applied to a model see :ref:`here<modeling_transformations>`.


Estimate model
~~~~~~~~~~~~~~

When we have created our model we can perform an estimation using ``fit``:

.. pharmpy-code::
    from pharmpy.tools import fit

    res = fit(model)

.. note::

    In order to esimate using any of the supported softwares (NONMEM, nlmixr2, rxode2) you need to have a configuration
    file set up with a path to NONMEM, instructions can be found :ref:`here <config_page>`.

Analyze the results
~~~~~~~~~~~~~~~~~~~

We can now analyze the fit results. 

.. pharmpy-execute::
   :hide-code:
   :hide-output:

    from pharmpy.tools import read_results
    
    res = read_results('tests/testdata/results/results_example1.json')

Let's look at the ofv:

.. pharmpy-execute::
    res.ofv

and the parameter estimates:

.. pharmpy-execute::
    res.parameter_estimates

For more information about model estimation see :ref:`here<model_estimation_and_results>`.
