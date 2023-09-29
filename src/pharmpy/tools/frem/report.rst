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
