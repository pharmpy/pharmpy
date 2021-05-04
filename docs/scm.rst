===
scm
===

Pharmpy currently creates results after a PsN scm run.

~~~~~~~~~~~~~~~
The scm results
~~~~~~~~~~~~~~~

Steps
~~~~~

The `steps` table contains information on the models run in each step.

.. jupyter-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools.scm.results import psn_scm_results
    res = psn_scm_results(pathlib.Path('tests/testdata/nonmem/scm/mergeofv_dir1'))
    res.steps

The table index has multiple levels

==============  =============================================
Level           Description
==============  =============================================
step            A step index starting with 1
parameter       Parameter to which covariate effect was added
covariate       Covariate used
extended_state  The covariate effect (state) added
==============  =============================================

The table columns are

====================  ==========================================================================================
Column                Description
====================  ==========================================================================================
reduced_ofv           OFV of the reduced model. Note that in the backwards direction this is not the base model.
extended_ofv          OFV of the extended model.
ofv_drop              Drop in OFV beween the reduced and extended models.
delta_df              The number of added population parameters in the extended model.
pvalue                p-value for the covariate effect being significant
goal_pvalue           target p-value for the run
is_backward           True if run was in a backward step and False if not
extended_significant  Is the covariate effect in the extended model significant?
selected              Was the covariate effect in the step selected
directory             Path to the model
model                 Name of the model
covariate_effects     Listing the covariate effect values
====================  ==========================================================================================


OFV Summary
~~~~~~~~~~~

The `ofv_summary` method can be used to create a slightly different view of the runs. The included relations in the final model are
listed under the direction "Final included".

.. jupyter-execute::

    res.ofv_summary()


Candidate Summary
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    res.candidate_summary()
