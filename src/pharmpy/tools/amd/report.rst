.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


AMD Results
===========

Final model
~~~~~~~~~~~

.. altair-plot::
    :hide-code:

    from pharmpy.results import read_results

    res = read_results('results.json')
    res.final_model_dv_vs_ipred_plot


.. altair-plot::
    :hide-code:

    res.final_model_cwres_vs_idv_plot
