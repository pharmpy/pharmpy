.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: LGPL-3.0-or-later

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


VPC Results
===========

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')
    res.plot

