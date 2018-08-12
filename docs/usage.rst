=====
Usage
=====

The :class:`pysn.Model` class is a model agnostic entry point. For example, to
read the NONMEM model ``pheno_real.mod``, the following is sufficient:

.. testcode::
    :pyversion: > 3.6

    from pysn import Model

    path = 'tests/testdata/nonmem/pheno_real.mod'
    pheno = Model(path)

The API is automatically detected and used:

.. doctest::
    :pyversion: > 3.6

    >>> type(pheno)
    <class 'pysn.psn.api_nonmem.model.Model'>

Parsing uses lark-parser_ in the backend, with derived classes for some
automated tasks.

.. doctest::
    :pyversion: > 3.6

    >>> pheno.input.path.name
    'pheno.dta'
    >>> print(pheno.input.column_names())
    ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']
    >>> thetas = pheno.get_records('THETA')
    >>> thetas[0].root
    AttrTree(root) '  (0,0.00469307) ; CL\n'
    >>> str(thetas[0].root)
    '  (0,0.00469307) ; CL\n'
    >>> print(thetas[0].parser)
    root "  (0,0.00469307) ; CL\n"
     ├─ whitespace "  "
     │  └─ WS_MULTILINE "  "
     └─ theta "(0,0.00469307) ; CL\n"
        ├─ LP "("
        ├─ lower_bound "0"
        │  └─ NUMERIC "0"
        ├─ sep ","
        │  └─ COMMA ","
        ├─ init "0.00469307"
        │  └─ NUMERIC "0.00469307"
        ├─ RP ")"
        ├─ whitespace " "
        │  └─ WS " "
        └─ comment "; CL\n"
           ├─ SEMICOLON ";"
           ├─ text " CL"
           │  └─ NOT_NL " CL"
           └─ NL "\n"
    >>> for node in thetas[0].root.tree_walk():
    ...     print(node.__class__.__name__, node.rule, repr(str(node)))
    AttrTree whitespace '  '
    AttrToken WS_MULTILINE '  '
    AttrTree theta '(0,0.00469307) ; CL\n'
    AttrToken LP '('
    AttrTree lower_bound '0'
    AttrToken NUMERIC '0'
    AttrTree sep ','
    AttrToken COMMA ','
    AttrTree init '0.00469307'
    AttrToken NUMERIC '0.00469307'
    AttrToken RP ')'
    AttrTree whitespace ' '
    AttrToken WS ' '
    AttrTree comment '; CL\n'
    AttrToken SEMICOLON ';'
    AttrTree text ' CL'
    AttrToken NOT_NL ' CL'
    AttrToken NL '\n'

.. _lark-parser: https://pypi.org/project/lark-parser/
