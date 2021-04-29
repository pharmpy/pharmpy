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
    res.plot_parameter_estimates_histogram()

OFV distribution
~~~~~~~~~~~~~~~~

.. altair-plot::
    :hide-code:

    res.plot_ofv()

Parameter estimates correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_parameter_estimates_correlation()

Estimated degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. altair-plot::
   :hide-code:

   res.plot_dofv_quantiles()
