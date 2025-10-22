.. toctree::
   :maxdepth: 2
   :caption: Contents:

   results


QA Results
==========

Overview
--------

.. jupyter-execute::
    :hide-code:

    from pharmpy.workflows.results import read_results
    from pharmpy.visualization import display_table

    res = read_results('results.json')
    new_names = {'dofv': 'dOFV', 'added_params': 'Added parameters'}

    def _rename_model(name):
        if 'boxcox' in name:
            return 'Box-cox transformation'
        elif 'tdist' in name:
            return 't-distribution'
        elif 'fullblock' in name:
            return 'Full OMEGA Block'
        else:
            return name

    new_row_names = {name: _rename_model(name) for name in res.dofv.index.values}
    dofv = res.dofv.rename(columns=new_names, index=new_row_names)

    display_table(dofv, remove_nan_columns=False)


Parameter Variability Model
---------------------------

Box-cox transformation
^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::
    :hide-code:

    res.boxcox_plot

.. jupyter-execute::
    :hide-code:

    if res.boxcox_parameters is not None:
        new_names = {'lambda': 'Lambda', 'old_sd': 'Old SD', 'new_sd': 'New SD'}
        boxcox_parameters = res.boxcox_parameters.rename(columns=new_names)
        display_table(boxcox_parameters, remove_nan_columns=False)


t-distribution transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. jupyter-execute::
    :hide-code:

    res.tdist_plot

.. jupyter-execute::
   :hide-code:

    if res.tdist_parameters is not None:
        new_names = {'df': 'Degrees of freedom', 'old_sd': 'Old SD', 'new_sd': 'New SD'}
        tdist_parameters = res.tdist_parameters.rename(columns=new_names)
        display_table(tdist_parameters, remove_nan_columns=False)

Full OMEGA Block
^^^^^^^^^^^^^^^^

.. jupyter-execute::
    :hide-code:

    if res.fullblock_parameters is not None:
        new_names = {'new': 'New', 'old': 'Old'}
        fullblock_parameters = res.fullblock_parameters.rename(columns=new_names)
        display_table(fullblock_parameters, remove_nan_columns=False)
