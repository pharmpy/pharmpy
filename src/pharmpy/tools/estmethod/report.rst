.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: LGPL-3.0-or-later

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

Estmethod Results
=================


Settings
~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')
    res.summary_settings


Result summary
~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.summary_tool
