.. _design-section:

======
Design
======

Introduction
============

:mod:`PysN <pysn>` is structured into a generic *template*, defined in :mod:`pysn.generic`, and
*implementations* of that template. The implementations define the API, but are guided and assisted
by the generic template.

A NONMEM7 development implementation (top at :mod:`pysn.api_nonmem`) is *the* implementation so far
and will thus stand-in for *any* implementation in this document.

The Heart, The :class:`~pysn.generic.Model` Class
=================================================

The main (and only, so far) entry point is :func:`pysn.model.Model`, which abstracts the particular
model class call and returns the object.

Basic ideas:

- :class:`Generic <pysn.generic.Model>` template is model (type) *agnostic*
- :class:`Implementation <pysn.api_nonmem.model.Model>` is *not agnostic*.
- No entanglement with output and runs, just the model definition
- Implementation detects support (to get the job) and parses one type of model
- Duck typing is the aim. However, an implementation *must* inherit the generic template to guide consistency in implementation and cause fallthrough to ``NotImplementedError``.

The Model APIs
==============

.. note:: Thou shalt not bloat the Model class.

A pharmacometric model and workflow has many facets. APIs are abstract concepts like operations on
models, of which these exist thus far:

+---------------------------+-------------------------------------+--------------------------------------------+
| Template (generic)        | NONMEM7 (implementation)            | Notes                                      |
+===========================+=====================================+============================================+
| :class:`~pysn.generic`    | :class:`~pysn.api_nonmem`           | Top-level module                           |
+---------------------------+-------------------------------------+--------------------------------------------+
| :class:`~pysn.model`      | :class:`~pysn.api_nonmem.model`     | Binds everything else together             |
+---------------------------+-------------------------------------+--------------------------------------------+
| **WIP**                   | :class:`~pysn.api_nonmem.input`     | Model input (the data)                     |
+---------------------------+-------------------------------------+--------------------------------------------+
| :class:`~pysn.output`     | :class:`~pysn.api_nonmem.output`    | Model output (estimates, etc.)             |
+---------------------------+-------------------------------------+--------------------------------------------+
| :class:`~pysn.parameters` | :class:`~pysn.api_nonmem.parameters`| Parameter model abstraction                |
+---------------------------+-------------------------------------+--------------------------------------------+
| :class:`~pysn.execute`    | :class:`~pysn.api_nonmem.execute`   | Execution of model                         |
+---------------------------+-------------------------------------+--------------------------------------------+
|                           | :class:`~pysn.api_nonmem.detect`    | Detection of model support                 |
+---------------------------+-------------------------------------+--------------------------------------------+
|                           | :class:`~pysn.api_nonmem.records`   | Non-agnostic implementation detail example |
+---------------------------+-------------------------------------+--------------------------------------------+

**API module: :mod:`~pysn.execute`**

.. note:: This needs some more thought.

Is comprised of

- :class:`~pysn.execute.job.Job` A job. Can contain several non-blocking executions (e.g. bootstrap, SIR, etc.).
- :class:`~pysn.execute.engine.Engine` Creates run command and stuff. E.g. to start a job with ``nmfe``.
- :class:`~pysn.execute.environment.Environment` Is the cluster or local or OS etc to start jobs on.
- :class:`~pysn.execute.run_directory.RunDirectory` Run directory, invoking directory, where are the models? Which files to copy where? Organization of files.

**API module: :mod:`~pysn.input`**

.. warning:: This is outdated. Fix this!

One central dataset storage implementation. Different interpretations of different columns are
needed like ``EVID``, ``AMT`` etc.

**API module: :mod:`~pysn.output`**

.. note:: This needs some more thought.

Read in one type of output and convert to SO or other standardised output storage.

**API module: :mod:`~pysn.transform`**

Transforms models. Should comprise collection of functions that generally take model as input, apply
changes (no copy) and returns it. E.g. adding covariances to expand covariance matrices, changing
distributions (e.g. Box-Cox), etc. Even updating initials is likely to end up here.

.. note:: No implicit disk writes. Thank you.

**API module: :mod:`~pysn.tool`**

A bundle of operations. Organizes a run with standard files generated (like `meta.yaml`?).

Design & Ideas
==============

Second-layer abstraction
------------------------

Shall be generally followed throughout, where a 2nd layer bridges the
non-agnostic implementation details to the agnostic shared functionality (and ultimately, the tools
using the API). Such a 2nd layer shall have these characteristics:

1. Define a generic template with functions swallowing as much as possible, with ``raise
   NotImplementedError`` where such cannot be done.
2. Be inherited by implementations for specificity. Ideally, ``super()`` shall be used as much as
   possible (*especially* in ``__init__``).
3. Generic templates hold helper classes that *mustn't* be inherited. Unless it's a good idea
   somewhere, but I doubt it (so forget it). These contain data which has been extracted, is
   bi-directional and can be applied across model types.

Example is :mod:`~pysn.parameters` with the generic API
:class:`~pysn.parameters.parameter_model.ParameterModel`
(:class:`~pysn.api_nonmem.parameters.parameter_model.ParameterModel` implements), but also these 2nd
layers:

- :class:`~pysn.parameters.parameter_model.distributions.CovarianceMatrix`
- :class:`~pysn.parameters.parameter_model.scalars.Covar`
- :class:`~pysn.parameters.parameter_model.scalars.Scalar`
- :class:`~pysn.parameters.parameter_model.scalars.Var`
- :class:`~pysn.parameters.parameter_model.vectors.ParameterList`

Usage of these shall not require any knowledge of the implementation. It is
return value of e.g.::

   model.parameters.inits()  # generates ParameterModel and returns ParameterList

No Caching
----------

Well, not more than *necessary*. This means that ``model.parameters`` above, generates the
``ParameterModel`` object at request. In *THE implementation*, this is through requesting data from
:class:`~pysn.api_nonmem.records.theta_record.ThetaRecord`,
:class:`~pysn.api_nonmem.records.theta_record.OmegaRecord` (which uses 2nd layers in this case). All
this is then bound into a :class:`~pysn.parameters.parameter_model.vectors.ParameterList` (which
inherits :class:`list`) and returned.

Inherit Base Types
------------------

If the 2nd layer is e.g. "list-like", just inherit :class:`list`. It's Python 3 and it's all good!


Why Multiple APIs?
------------------

Why multiple APIs in a hierarchy and not only one directly on the model class? Compare code::

   model.input.column_list()

with::

   # this pollutes the poor namespace!
   model.column_list()
