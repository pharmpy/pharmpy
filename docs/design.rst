.. _design-section:

======
Design
======

Introduction
============

:mod:`PharmPy <pharmpy>` is structured into a generic *template*, defined in :mod:`pharmpy.generic`, and
*implementations* of that template. The implementations define the API, but are guided and assisted
by the generic template.

A NONMEM7 development implementation (top at :mod:`pharmpy.api_nonmem`) is *the* implementation so far
and will thus stand-in for *any* implementation in this document.

The Heart, The :class:`~pharmpy.generic.Model` Class
====================================================

The main (and only, so far) entry point is :func:`pharmpy.model.Model`, which abstracts the particular
model class call and returns the object.

Basic ideas:

- :class:`Generic <pharmpy.generic.Model>` template is model (type) *agnostic*
- :class:`Implementation <pharmpy.api_nonmem.model.Model>` is *not agnostic*.
- No entanglement with output and runs, just the model definition
- Implementation detects support (to get the job) and parses one type of model
- Duck typing is the aim. However, an implementation *must* inherit the generic template to guide consistency in implementation and cause fallthrough to ``NotImplementedError``.

The Model APIs
==============

.. note:: Thou shalt not bloat the Model class.

A pharmacometric model and workflow has many facets. APIs are abstract concepts like operations on
models, of which these exist thus far:

   .. list-table::
       :widths: 25 25 30
       :header-rows: 1

       - - Template (generic)
         - NONMEM7 (implementation)
         - Notes

       - - :mod:`~pharmpy.generic`
         - :mod:`~pharmpy.api_nonmem`
         - Top-level module

       - - :mod:`~pharmpy.model`
         - :mod:`~pharmpy.api_nonmem.model`
         - Binds everything else together

       - - **WIP**
         - :mod:`~pharmpy.api_nonmem.input`
         - Model input (the data)

       - - :mod:`~pharmpy.output`
         - :mod:`~pharmpy.api_nonmem.output`
         - Model output (estimates, etc.)

       - - :mod:`~pharmpy.parameters`
         - :mod:`~pharmpy.api_nonmem.parameters`
         - Parameter model abstraction

       - - :mod:`~pharmpy.execute`
         - :mod:`~pharmpy.api_nonmem.execute`
         - Execution of model

       - -
         - :mod:`~pharmpy.api_nonmem.detect`
         - Detection of model support

       - -
         - :mod:`~pharmpy.api_nonmem.records`
         - Non-agnostic implementation detail example

API module: :mod:`~pharmpy.execute`
-----------------------------------

.. note:: This needs some more thought (from someone who isn't me).

See :mod:`the package <pharmpy.execute>` and its modules (and their classes) for technical information.

This package defines four classes. These are my (module documentation non-overlapping) thoughts
about them.

:class:`~pharmpy.execute.job.Job` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Will need to inherit where `~pharmpy.execute.environment.Environment` does.

:class:`~pharmpy.execute.run_directory.RunDirectory` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Will not need non-generic extensions (as I see it).

If no parent directory given, temporary directory shall be created (and destroyed). There shall not
be any implicit usages of working directories or special meaning attributed to :attr:`Model.path
<pharmpy.generic.Model.path>`.

Purpose is to contain input/output files and the execution process of a "execution-like task" of
a model. Thus, it has an attribute, :attr:`RunDirectory.model
<pharmpy.execute.run_directory.RunDirectory.model>` which "takes" a model via performing a deepcopy if
not in directory already, and changing :attr:`Model.path <pharmpy.generic.Model.path>`. That `path`
change should then trigger a re-pathing of the intra-model (relative) paths (e.g. to data).

   .. todo::
      Changing :attr:`Model.path <pharmpy.generic.Model.path>` should trigger changes to all contained
      relative paths.

:class:`~pharmpy.execute.run_directory.RunDirectory` is a **sandbox** for execution, even if trivially
escapable. Thus, it should also provide methods for file operations that ensure the operation is
contained.

   .. todo::
      Develop "safe" file operation methods for :class:`~pharmpy.execute.run_directory.RunDirectory`.

Class shall expose an API that can be used to copy back the resulting (output) files, just as PsN
does, but it *must* be requested explicitly. If not used when run directory is temp directory, this
*must* guarantee loss of files. The copy will occur whenever requested or in
:func:`pharmpy.execute.run_directory.RunDirectory.__del__`, just as file deletion works already.

:class:`~pharmpy.execute.environment.Environment` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Needs an implementation per OS (Posix/Windows) and per system (system, SLURM, etc.).
- Execution without SLURM etc. is called "system execution".

:class:`~pharmpy.execute.engine.Engine` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Creates Job objects, executing some task, for a *specific* implementation.
- Contains Environment object. The focal point for implementation inheritance.

API module: :mod:`~pharmpy.input`
---------------------------------

.. warning:: This is outdated. Fix this!

One central dataset storage implementation. Different interpretations of different columns are
needed like ``EVID``, ``AMT`` etc.

API module: :mod:`~pharmpy.output`
----------------------------------

.. note:: This needs some more thought.

Read in one type of output and convert to SO or other standardised output storage.

API module: :mod:`~pharmpy.transform`
-------------------------------------

.. warning:: Planned but not yet started.

Transforms models. Should comprise collection of functions that generally take model as input, apply
changes (no copy) and returns it. E.g. adding covariances to expand covariance matrices, changing
distributions (e.g. Box-Cox), etc. Even updating initials is likely to end up here.

.. note:: No implicit disk writes. Thank you.

API module: :mod:`~pharmpy.tool`
--------------------------------

.. warning:: Planned but not yet started.

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

Example is :mod:`~pharmpy.parameters` with the generic API
:class:`~pharmpy.parameters.parameter_model.ParameterModel`
(:class:`~pharmpy.api_nonmem.parameters.parameter_model.ParameterModel` implements), but also these 2nd
layers:

- :class:`~pharmpy.parameters.parameter_model.distributions.CovarianceMatrix`
- :class:`~pharmpy.parameters.parameter_model.scalars.Covar`
- :class:`~pharmpy.parameters.parameter_model.scalars.Scalar`
- :class:`~pharmpy.parameters.parameter_model.scalars.Var`
- :class:`~pharmpy.parameters.parameter_model.vectors.ParameterList`

Usage of these shall not require any knowledge of the implementation. It is
return value of e.g.::

   model.parameters.inits()  # generates ParameterModel and returns ParameterList

No Caching
----------

Well, not more than *necessary*. This means that ``model.parameters`` above, generates the
``ParameterModel`` object at request. In *THE implementation*, this is through requesting data from
:class:`~pharmpy.api_nonmem.records.theta_record.ThetaRecord`,
:class:`~pharmpy.api_nonmem.records.theta_record.OmegaRecord` (which uses 2nd layers in this case). All
this is then bound into a :class:`~pharmpy.parameters.parameter_model.vectors.ParameterList` (which
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
