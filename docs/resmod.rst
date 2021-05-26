======
Resmod
======

Pharmpy currently creates results after a PsN resmod model.

~~~~~~~~~~~~~~~~~~
The resmod results
~~~~~~~~~~~~~~~~~~

Resmod models
~~~~~~~~~~~~~

The `models` table contains a summary of all resmod models

.. jupyter-execute::
    :hide-code:

    import pathlib
    from pharmpy.tools.resmod.results import psn_resmod_results
    res = psn_resmod_results(pathlib.Path('tests/testdata/psn/resmod_dir1'))
    res.models
