=====
Usage
=====

The :class:`pysn.Model` class is a model agnostic entry point. For example, to
read the NONMEM model ``pheno_real.mod`` the following is sufficient:

.. testcode::
    :pyversion: > 3.6

    from pysn import Model

    path = 'tests/testdata/nonmem/pheno_real.mod'
    pheno = Model(path)

The format is automatically detected:

.. doctest::
    :pyversion: > 3.6

    >>> type(pheno)
    <class 'pysn.psn.api_nonmem.model.Model'>

Parsing is only performed when necessary, e.g.

.. doctest::
    :pyversion: > 3.6

    >>> pheno.input.path.name
    'pheno.dta'
    >>> print(pheno.input.column_names())
    ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    >>> thetas = pheno.get_records('THETA')
    >>> for token in thetas[0].tokens:
    ...     print(token.type, repr(str(token.content)))
    ThetaTokenType.WHITESPACE '  '
    ThetaTokenType.OPENPAREN '('
    ThetaTokenType.TOKEN '0'
    ThetaTokenType.COMMA ','
    ThetaTokenType.TOKEN '0.00469307'
    ThetaTokenType.CLOSEPAREN ')'
    ThetaTokenType.WHITESPACE ' '
    ThetaTokenType.COMMENT '; CL'
    ThetaTokenType.WHITESPACE '\n'
