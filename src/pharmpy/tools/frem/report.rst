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

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.plot_covariate_effects()


Individual effects
~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_individual_effects()

Unexplained variability
~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_unexplained_variability()
