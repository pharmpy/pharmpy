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
-------------------

.. jupyter-execute::
   :hide-code:

   from pharmpy.visualization import display_table

   display_table(final_model_parameter_estimates, format={'estimates': '{:,.4f}', 'RSE': '{:,.1%}'})



Eta shrinkage
-------------

.. jupyter-execute::
   :hide-code:

   display_table(res.final_model_eta_shrinkage.to_frame(name='eta shrinkage') , format={'eta shrinkage': '{:,.4f}'})

CWRES vs TIME
-------------

.. jupyter-execute::
    :hide-code:

    res.final_model_cwres_vs_idv_plot


DV vs PRED
----------

.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_pred_plot

DV vs IPRED
-----------

.. jupyter-execute::
   :hide-code:

   res.final_model_dv_vs_ipred_plot


ETA distribution
----------------

.. jupyter-execute::
   :hide-code:

   res.final_model_eta_distribution_plot

Tool summary
~~~~~~~~~~~~

.. jupyter-execute::
   :hide-code:

   display_table(res.summary_tool)

Model summary
~~~~~~~~~~~~~

.. jupyter-execute::
   :hide-code:

   display_table(res.summary_models)
