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
