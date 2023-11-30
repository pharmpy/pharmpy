.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


AMD Results
===========

Final model
~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results

    res = read_results('results.json')


.. jupyter-execute::
   :hide-code:

   res.final_model_parameter_estimates.style.format({
       'estimates': '{:,.4f}'.format,
       'RSE': '{:,.1%}'.format,
   })


.. jupyter-execute::
    :hide-code:

    res.final_model_cwres_vs_idv_plot


.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_pred_plot

  
.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_ipred_plot
