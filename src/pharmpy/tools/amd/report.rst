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
-------------------

.. jupyter-execute::
   :hide-code:

   from pharmpy.visualization import display_table

   display_table(res.final_model_parameter_estimates, format={'estimates': '{:,.4f}', 'RSE': '{:,.1%}'})



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
