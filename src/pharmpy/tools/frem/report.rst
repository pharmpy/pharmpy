.. SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
..
.. SPDX-License-Identifier: LGPL-3.0-or-later

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


FREM Results
============

Covariate effects
~~~~~~~~~~~~~~~~~


.. altair-plot::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')
    res.covariate_effects_plot


Individual effects
~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.individual_effects_plot

Unexplained variability
~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.unexplained_variability_plot
