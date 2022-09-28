.. figure:: Pharmpy_logo.svg
   :class: only-light
   :width: 400
   :align: center

.. figure:: Pharmpy_logo_dark.svg
   :class: only-dark
   :width: 400
   :align: center

==================
Welcome to Pharmpy
==================


Pharmpy is a library and a set of tools for pharmacometrics. It can be used as a regular Python package, in R using
pharmr package or via its built-in command line interface. The API of Pharmpy is model agnostic and
the library is architectured to be able to handle different types of model and data formats (currently
mainly NONMEM is supported, but some support for nlmixr is available).

Current features include:

* A model abstraction as a foundation for higher level operations
* Functions to transform models and datasets, extract information from models and read and calculate results
* Parsing of NONMEM models, datasets and result files
* Generating updated NONMEM code for modified models
* Complex tools to for example optimize the iiv structure or residual error of a model
* CLI supporting operations on datasets and models and running tools

We encourage your contribution to the Pharmpy project! You can report issues and post suggestions of features via
GitHub issues (to `Pharmpy <https://github.com/pharmpy/pharmpy/issues>`_ or to
`pharmr <https://github.com/pharmpy/pharmr/issues>`_). If you want to contribute with code you can find more information
under :ref:`contribute`.


Pharmpy is maintained by the Uppsala University Pharmacometrics group and is an open-source project.

.. toctree::
   :hidden:

   About us <contributors>
   getting_started
   user_guide
   api
   changelog
   contribute
   References and citation <citation>
   license
   developers
