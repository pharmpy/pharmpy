.. _model:

=================
The Pharmpy model
=================

At the heart of Pharmpy lies its non-linear mixed effects model abstraction. A model needs to follow the interface of
the base model class :py:class:`pharmpy.model.Model`. This means that any model format can in theory be supported by
Pharmpy via subclasses that implement the same base interface. This makes any operation performed on a model be the
same regardless of the underlying implementation of the model and it is one of the core design principles of Pharmpy.
This allows the functions in :py:mod:`pharmpy.modeling` and tools in :py:mod:`pharmpy.tools` to be independent
on the estimation software: it is only when writing and fitting a model that the implementation is estimation software
specific.

.. warning::

    This section is intended to give an overview of the Pharmpy model and its different components. For higher level
    manipulations, please check out the functions in :py:mod:`pharmpy.modeling`, which are described in further detail
    in :ref:`the modeling user guide<modeling>`


~~~~~~~~~~~~~~~~~~
Reading in a model
~~~~~~~~~~~~~~~~~~

Reading a model from a model specification file into a Pharmpy model object can be done by using the
:py:func:`pharmpy.modeling.read_model` function.

.. pharmpy-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. pharmpy-execute::
   :hide-output:

   from pharmpy.modeling import read_model

   model = read_model(path / "pheno_real.mod")


Internally this will trigger a model type detection to select which model implementation to use, i.e. if it is an
NM-TRAN control stream the Pharmpy NONMEM model subclass will be selected.

.. _model_write:

~~~~~~~~~~~~~~~
Writing a model
~~~~~~~~~~~~~~~

A model object can be written to a file using its source format. By default the model file will be created in the
current working directory using the name of the model.

.. pharmpy-code::

    from pharmpy.modeling import write_model
    write_model(model)

Optionally a path can be specified:

.. pharmpy-code::

   write_model(model, path='/home/user/mymodel.mod')


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inspecting the model attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Name and description
~~~~~~~~~~~~~~~~~~~~

A model has a name property that can be read or changed. After reading a model from a file the name is set to the
filename without extension.

.. pharmpy-execute::

   model.name



Parameters
~~~~~~~~~~

Model parameters are scalar values that are used in the mathematical definition of the model and are estimated when a
model is fit from data. The parameters of a model are thus optimization parameters and can in turn be used as
parameters in statistical distributions or as structural parameters. A parameter is represented by using the
:py:class:`pharmpy.model.Parameter` class.

It is often convenient to work with a set of parameters at the same time, for example all parameters of a model.
In Pharmpy multiple parameters are organized using the :py:class:`pharmpy.model.Parameters` class as an ordered set of
:py:class:`pharmpy.model.Parameter`. All parameters of a model can be accessed by using the parameters attribute:

.. pharmpy-execute::

   parset = model.parameters
   parset

Operations on multiple parameters are made easier using methods or properties on parameter sets. For example, to get
all initial estimates as a dictionary:

.. pharmpy-execute::

   parset.inits

Each parameter can be retrieved using indexing:

.. pharmpy-execute::

   par = parset['PTVCL']

A model parameter must have a name and an initial value and can optionally be constrained to a lower and or upper bound.
A parameter can also be fixed meaning that it will be set to its initial value. The parameter attributes can be read
out via properties.

.. pharmpy-execute::

   par.lower

Random variables
~~~~~~~~~~~~~~~~

The random variables of a model are available through the ``random_variables`` property and are organized using the
:py:class:`pharmpy.model.RandomVariables` which is an ordered set of distributions of either
:py:class:`pharmpy.model.NormalDistribution` or :py:class:`pharmpy.model.JointNormalDistribution` class. All random
variables of a model can be accessed by using the random variables attribute:

.. pharmpy-execute::

   rvs = model.random_variables
   rvs

The set of random variables can be split into subsets of random variables, for example IIVs:

.. pharmpy-execute::

   rvs.iiv

A distribution can be extracted using the name of one of the etas:

.. pharmpy-execute::

   dist = rvs['ETA_1']
   dist

Similarly to parameters, we can extract different attributes from the distribution:

.. pharmpy-execute::

   dist.names

Statements
~~~~~~~~~~

The model statements represent the mathematical description of the model. All statements can be retrieved via the
statements property as a :py:class:`pharmpy.model.Statements` object, which is a list of model statements of either the
class :py:class:`pharmpy.model.Assignment` or :py:class:`pharmpy.model.ODESystem`.

.. pharmpy-execute::

   statements = model.statements
   statements

Changing the statements of a model can be done by setting the statements property. This way of manipulating a model is
quite low level and flexible but cumbersome. For higher level model manipulation use the :py:mod:`pharmpy.modeling`
module.

If the model has a system of ordinary differential equations this will be part of the statements. It can easily be
retrieved from the statement object

.. pharmpy-execute::

   statements.ode_system

Get the amounts vector:

.. pharmpy-execute::

   statements.ode_system.amounts

Get the compartmental matrix:

.. pharmpy-execute::

   statements.ode_system.compartmental_matrix

Dataset and datainfo
~~~~~~~~~~~~~~~~~~~~

See :ref:`dataset`.

Estimation steps
~~~~~~~~~~~~~~~~

The :py:class:`pharmpy.model.EstimationSteps` object contains information on how to estimate the model.

.. pharmpy-execute::

   ests = model.estimation_steps
   ests

Dependent variables
~~~~~~~~~~~~~~~~~~~

A model can describe one or more dependent variables (output variables). Each dependent variable is defined in the
``dependent_variables`` attribute. This is a dictionary of each dependent variable symbol to the corresponding ``DVID``.
If there is only one dependent variable the ``DVID`` column in the dataset is not needed and its value in this
definition is unimportant. The expressions of the dependent variables are all found in the statements.

.. pharmpy-execute::

    model.dependent_variables

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Low level manipulations of the model object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating and adding parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to create a parameter of the :py:class:`pharmpy.model.Parameter` class.

.. pharmpy-execute::

   from pharmpy.model import Parameter

   par = Parameter('THETA_1', 0.1, upper=2, fix=False)
   par

Substituting symbolic random variable distribution with numeric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. pharmpy-execute::

   from pharmpy.tools import read_modelfit_results

   frem_path = path / "frem" / "pheno" / "model_4.mod"
   frem_model = read_model(frem_path)
   frem_model_results = read_modelfit_results(frem_path)

   rvs = frem_model.random_variables
   rvs

Starting by extracting the variance:

.. pharmpy-execute::

   omega = rvs['ETA_1'].variance
   omega

Substitution of numerical values can be done directly from initial values

.. pharmpy-execute::

   omega.subs(frem_model.parameters.inits)

or from estimated values

.. pharmpy-execute::

   omega_est = omega.subs(dict(frem_model_results.parameter_estimates))
   omega_est

Operations on this parameter matrix can be done either by using SymPy

.. pharmpy-execute::

   omega_est.cholesky()

or in a pure numerical setting in NumPy

.. pharmpy-execute::

   import numpy as np

   a = np.array(omega_est).astype(np.float64)
   a

.. pharmpy-execute::

   np.linalg.cholesky(a)
