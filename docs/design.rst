.. _design:

=================
Design principles
=================


.. warning::
    This information is under construction and not complete

Levels of the architecture
==========================

The Pharmpy architecture has four distinct layers:

1. Plugins supporting various modeling languages and external estimation tools
2. The Model class
3. The modeling functional module
4. The worklows and tools

The Heart, The :class:`~pharmpy.model.Model` Class
==================================================

The Model class is the central abstraction in Pharmpy. It represents a non-linear mixed effects model. Each component of a model
is represented by its own class that can be handled separately from the model. Classes are designed so that they have a minimum of
methods. Higher level operations that can use the public interface of a class are instead put in the modeling module. This is to decrease
coupling. A model defines multiple mathematical symbols and all are concidered to live together in one single namespace.

:class:`pharmpy.Parameters`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``MutableSequence`` of scalar real optimization parameters. Parameters can be constrained to real intervals, but higher level constraints
such as positive semidefiniteness of parameters used as elements of covariance matrices does not concern this class.

:class:`pharmpy.RandomVariables`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``MutableSequence`` of random variables (etas and epsilons). 

:class:`pharmpy.Statements`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``MutableSequence`` of model statements. Statements can be 


The :py:mod:`pharmpy.modeling` module
=====================================

This module contains functions operating on lower level objects, mostly models. Functions here are exposed directly to users
of ``pharmpy.modeling`` and to users of the ``pharmr`` R wrapper package so docstrings should keep a high standard using ``See also``
and ``Examples`` for most functions. 

Docstrings
==========

All docstrings follow the numpy standard. See https://numpydoc.readthedocs.io/en/latest/format.html

Dependencies
============

Three of the Pharmpy dependences have special status. These are sympy, pandas and numpy. These packages are considered
to be central in the Python scientific ecosystem. Thus exposing objects from classes defined in these packages is ok
and encouraged in the Pharmpy API. This will make it easier for users to build on Pharmpy and removes the need for wrapping
these object into the Pharmpy API. Since symengine objects share API with sympy exposing symengine objects are also ok.

Classes and APIs for all other dependencies should not be exposed to users to avoid leaking third party APIs. This will
simplify replacing dependencies without breaking the Pharmpy API.
