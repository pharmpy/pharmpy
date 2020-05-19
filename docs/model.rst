=================
The Pharmpy model
=================

At the heart of Pharmpy lies its non-linear mixed effects model abstraction. A model needs to follow the interface of the base model class :py:class:`pharmpy.model.Model`. This means that any model format can in theory be supported by Pharmpy via subclasses that implement the same base interface. This makes any operation performed on a model be the same regardless of the underlying implementation of the model and it is one of the core design principles of Pharmpy.


.. math::

~~~~~~~~~~~~~~~~~~
Reading in a model
~~~~~~~~~~~~~~~~~~

Reading a model from a model specification file into a Pharmy model object is done by using the factory constructor

.. jupyter-execute::
   :hide-output:
   :hide-code:

   from pathlib import Path
   path = Path('tests/testdata/nonmem/')

.. jupyter-execute::

   from pharmpy import Model

   model = Model(path / "pheno_real.mod")


Internally this will trigger a model type detection to select which model implementation to use, i.e. if it is an NM-TRAN control stream the Pharmpy NONMEM model class will be selected transparent to the user. 

~~~~~~~~~~
Model name
~~~~~~~~~~

A model has a name property that can be read or changed. After reading a model from a file the name is set to the filename without extension. 

.. jupyter-execute::

   model.name


~~~~~~~~~~~~~~~
Writing a model
~~~~~~~~~~~~~~~

A model object can be written to a file using its source format. By default the model file will be created in the current working directory using the name of the model.

.. code-block::

   model.write()

Optionally a path can be specified:

.. code-block::

   model.write(path='/home/user/mymodel.mod')


~~~~~~~~~~~~~~~~~~~~~~
Getting the model code
~~~~~~~~~~~~~~~~~~~~~~

The model code can be retrieved as a string directly by the string method of the model object.

.. jupyter-execute::

   print(model)

~~~~~~~~~~~~~~~~
Model parameters
~~~~~~~~~~~~~~~~

Model parameters are scalar values that are used in the mathematical definition of the model and are estimated when a model is fit from data. The parameters of a model are thus optimization parameters and can in turn be used as parameters in statistical distributions or as structural parameters. A parameter is defined using the :py:class:`pharmpy.Parameter` class.

.. jupyter-execute::

   from pharmpy import Parameter

   par = Parameter('THETA(1)', 0.1, upper=2, fix=False)
   par

A model parameter must have a name and an inital value and can optionally be constrained to a lower and or upper bound. A parameter can also be fixed meaning that it will be set to its initial value. The parameter attributes can be read out or changed via properties.

.. jupyter-execute::

   par.lower = -1
   print(par)

The parameter space of a parameter can be retrieved via a property:

.. jupyter-execute::

      par.parameter_space

~~~~~~~~~~~~~~
Parameter sets
~~~~~~~~~~~~~~

It is often convenient to work with a set of parameters at the same time, for example all parameters of a model. In Pharmpy a multiple parameters are organized in the :py:class:`pharmpy.ParameterSet` class as an ordered set of :py:class:`pharmpy.Parameter`. All parameters of a model can be accessed by using the parameters attribute:

.. jupyter-execute::

   parset = model.parameters
   parset

Each parameter can be retrieved using indexing

.. jupyter-execute::

   parset['THETA(1)']

Operations on multiple parameters are made easier using methods or properties on parameter sets. For example:

Get all initial estimates as a dictionary:

.. jupyter-execute::

   parset.inits

Setting initial estimates of some of the parameters:

.. jupyter-execute::

   parset.inits = {'THETA(1)': 0.5, 'OMEGA(1,1)': 0.05}
   parset

Fix some parameters:

.. jupyter-execute::

   parset.fix = {'THETA(2)': True, 'THETA(3)': True}
   parset

To update the parameter set of a model simply assign to the model property:

.. jupyter-execute::

   model.parameters = parset


~~~~~~~~~~~~~~~~
Modelfit results
~~~~~~~~~~~~~~~~

If a model has been fit the results can be retrieved directly from the model object. Here are some examples of the results that can be available:

.. jupyter-execute::

   model.modelfit_results.parameter_estimates


.. jupyter-execute::

   model.modelfit_results.covariance_matrix

.. jupyter-execute::

   model.modelfit_results.standard_errors

~~~~~~~~~~~~~~~~~~~~~~~~~~
Updating initial estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Updating all initial estimates of a model from its own results can be done either by directly setting:

.. jupyter-execute::

   model.parameters = model.modelfit_results.parameter_estimates

or using the convenience method:

.. jupyter-execute::

   model.update_inits()

~~~~~~~~~~~~~~~~
Random variables
~~~~~~~~~~~~~~~~

The random variables of a model are available through the random_variables property: 

.. jupyter-execute::

   rvs = model.random_variables
   rvs

Each random variable is a SymPy random variable and can be accessed separately using indexing:

.. jupyter-execute::

   eta1 = rvs['ETA(1)']

And the parameters of the random variable can be retrieved:

.. jupyter-execute::

   eta1.pspace.distribution.mean

.. jupyter-execute::

   eta1.pspace.distribution.std

Joint distributions are also supported

.. jupyter-execute::

   frem_model = Model(path / "frem" / "pheno" / "model_4.mod")

   rvs = frem_model.random_variables
   rvs

.. jupyter-execute::

   omega = rvs['ETA(1)'].pspace.distribution.sigma
   omega

Substitution of numerical values can be done directly from initial values

.. jupyter-execute::

   omega.subs(frem_model.parameters.inits)

or from estimated values

.. jupyter-execute::

   omega_est = omega.subs(dict(frem_model.modelfit_results.parameter_estimates))
   omega_est

Operations on this parameter matrix can be done either by using SymPy

.. jupyter-execute::

   omega_est.cholesky()

or in a pure numerical setting in NumPy

.. jupyter-execute::

   import numpy as np

   a = np.array(omega_est).astype(np.float64)
   a

.. jupyter-execute::

   np.linalg.cholesky(a)   
