.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


Bootstrap Results
=================

Distribution of parameter estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
    :hide-code:

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.parameter_estimates_histogram

OFV distribution
~~~~~~~~~~~~~~~~

.. altair-plot::
    :hide-code:

    res.ofv_plot

Estimated degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.dofv_quantiles_plot
