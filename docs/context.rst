.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _context:

=======
Context
=======

A context is where a tool is run, and it handles things like the model database and different general options.

+-------------------------------+-------------------------------------------------------------------------------------------------+
| Context                       | Description                                                                                     |
+===============================+=================================================================================================+
| :code:`LocalDirectoryContext` | This is the only context currently. Creates a directory locally with a local model database.    |
+-------------------------------+-------------------------------------------------------------------------------------------------+

A context can be opened and interacted with using:

.. pharmpy-code::

    from pharmpy.tools import open_context
    ctx = open_context('mycontext')

Name and ref
~~~~~~~~~~~~

The name and reference to the context specifies where the context is located.

Model database
~~~~~~~~~~~~~~

In :code:`models`, all model estimation files can be viewed, and the model database is where all model estimation results are
stored. All subcontexts will have its own :code:`models` directory but share model database with top level context.
See :ref:`model_database` for more information.

Annotations
~~~~~~~~~~~

A mapping between the name of a model in the tool (i.e. the names found in the tool result tables and in the :code:`models`
directory) and the name in the model database.

Log
~~~

Log of the tool run (includes warnings, errors, and abort messages). See :code:`broadcaster` in :ref:`options` to see
what types are supported for display of the log during the run.

Metadata
~~~~~~~~

A context will always contain a metadata file about how it was run. See :ref:`metadata` for more information.

Subcontexts
~~~~~~~~~~~

A context can have subcontexts. This happens when one tool calls another. They share a lot of features with a top level
context, like having a :code:`models` directory, some metadata, and annotations. A subcontext can also have its own
subcontexts. Unlike the top level tool, it doesn't have its own model database, dispatching options in the metadata,
nor the log.

