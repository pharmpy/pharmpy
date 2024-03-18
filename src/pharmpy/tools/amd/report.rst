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

Parameter estimates

.. jupyter-execute::
   :hide-code:

   if res.final_model_parameter_estimates['RSE'].any():
       results = res.final_model_parameter_estimates.style.format({
           'estimates': '{:,.4f}'.format,
           'RSE': '{:,.1%}'.format,
       })
   else:
       results = res.final_model_parameter_estimates['estimates'].to_frame(name='estimates').style.format({
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


.. jupyter-execute::
   :hide-code:

   res.final_model_vpc_plot
