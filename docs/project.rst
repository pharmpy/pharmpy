.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _project:

=======
Project
=======

A project is a way of managing tool runs. It can contain multiple contexts, which all share the same model database.

+-------------------------------+-------------------------------------------------------------------------------------------------+
| Context                       | Description                                                                                     |
+===============================+=================================================================================================+
| :code:`LocalDirectoryProject` | This is the only project currently. Creates a directory locally with a local model database.    |
+-------------------------------+-------------------------------------------------------------------------------------------------+

A project can be opened and interacted with using:

.. pharmpy-code::

    from pharmpy.tools import open_project
    proj = open_project('myproject')

It can then be used as in :ref:`options` for a tool run:

.. pharmpy-code::

    from pharmpy.modeling import load_example_model
    from pharmpy.tools import fit
    model = load_example_model('pheno')
    res = fit(model, project=proj)

Name and ref
~~~~~~~~~~~~

The name and reference to the project specifies where the project is located.

Model database
~~~~~~~~~~~~~~

The model database is where all model estimation results are stored. All contexts and their subcontexts will have their
own :code:`models` directory but share model database in the project. See :ref:`model_database` for more information.

