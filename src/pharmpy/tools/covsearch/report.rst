.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


COVsearch Results
==================

Final model
~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    from pharmpy.tools.common import table_final_parameter_estimates

    res = read_results('results.json')
    final_model_parameter_estimates = table_final_parameter_estimates(
            res.final_results.parameter_estimates_sdcorr,
            res.final_results.standard_errors_sdcorr
            )

Parameter estimates

.. jupyter-execute::
   :hide-code:

   if final_model_parameter_estimates['RSE'].any():
       results = final_model_parameter_estimates.style.format({
           'estimates': '{:,.4f}'.format,
           'RSE': '{:,.1%}'.format,
       })
   else:
       results = final_model_parameter_estimates['estimates'].to_frame(name='estimates').style.format({
           'estimates': '{:,.4f}'.format,
       })

   results



Eta shrinkage

.. jupyter-execute::
   :hide-code:

   res.final_model_eta_shrinkage.to_frame(name='eta shrinkage').style.format({
       'eta shrinkage': '{:,.4f}'.format,
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


.. jupyter-execute::
   :hide-code:

   res.final_model_eta_distribution_plot
