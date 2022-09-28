.. _plugins:

=======
Plugins
=======

Plugins are used to support reading and writing external model file formats, reading external results formats and running external estimation tools. This documentation is intened for developers wanting to understand the API of plugins to be able to write new plugins or maintain current plugins.

.. warning::
    This documentation is currently in construction.

Overview of a plugin
====================

A plugin is a Python module that exports certain functions. These functions will cover the tasks of detection, parsing, code generation and writing of models. All functions are not strictly necessesary and some functionality can be left out. A plugin could for example only support code generation and not parsing of a model file format. The functions are the following

+--------------+---------------------------------------------------+----------------+
| Name         | Operation                                         | Needed for     |
+==============+===================================================+================+
| detect_model | Detect if some model code is of this model format | Reading models | 
+--------------+---------------------------------------------------+----------------+
