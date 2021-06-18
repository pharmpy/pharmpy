==
QA
==

Pharmpy currently creates results after a PsN qa run.

~~~~~~~~~~~~~~
The qa results
~~~~~~~~~~~~~~

Overview
~~~~~~~~

.. jupyter-execute::
    :hide-code:

    import pathlib
    from pharmpy.modeling import read_results
    res = read_results(pathlib.Path('tests/testdata/results/qa_results.json'))
    res.dofv

Structural bias
~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.structural_bias

Boxcox
~~~~~~

.. jupyter-execute::
    :hide-code:

    res.boxcox_parameters

Tdist
~~~~~

.. jupyter-execute::
    :hide-code:

    res.tdist_parameters

Fullblock
~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.fullblock_parameters

Residual error
~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.residual_error

Covariate effects
~~~~~~~~~~~~~~~~~

.. jupyter-execute::
    :hide-code:

    res.covariate_effects
