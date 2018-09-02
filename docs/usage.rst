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
    <class 'pysn.api_nonmem.model.Model'>

Parsing uses lark-parser_ in the backend, with derived classes for some
automated tasks.

.. doctest::
    :pyversion: > 3.6

    >>> thetas = pheno.get_records('THETA')
    >>> thetas[0].root
    AttrTree('root', [AttrTree('ws', [AttrToken('WS_ALL', '  ')]), AttrTree('theta', [AttrToken('LPAR', '('), AttrTree('low', [AttrToken('NUMERIC', '0')]), AttrTree('sep', [AttrToken('COMMA', ',')]), AttrTree('init', [AttrToken('NUMERIC', '0.00469307')]), AttrToken('RPAR', ')')]), AttrTree('ws', [AttrToken('WS_ALL', ' ')]), AttrTree('comment', [AttrToken('SEMICOLON', ';'), AttrToken('WS', ' '), AttrToken('COMMENT', 'CL')]), AttrTree('ws', [AttrToken('WS_ALL', '\n')])])
    >>> str(thetas[0].root)
    '  (0,0.00469307) ; CL\n'
    >>> print(thetas[0].parser)
    root "  (0,0.00469307) ; CL\n"
     ├─ ws "  "
     │  └─ WS_ALL "  "
     ├─ theta "(0,0.00469307)"
     │  ├─ LPAR "("
     │  ├─ low "0"
     │  │  └─ NUMERIC "0"
     │  ├─ sep ","
     │  │  └─ COMMA ","
     │  ├─ init "0.00469307"
     │  │  └─ NUMERIC "0.00469307"
     │  └─ RPAR ")"
     ├─ ws " "
     │  └─ WS_ALL " "
     ├─ comment "; CL"
     │  ├─ SEMICOLON ";"
     │  ├─ WS " "
     │  └─ COMMENT "CL"
     └─ ws "\n"
        └─ WS_ALL "\n"
    >>> for node in thetas[0].root.tree_walk():
    ...     print(node.__class__.__name__, node.rule, repr(str(node)))
    AttrTree ws '  '
    AttrToken WS_ALL '  '
    AttrTree theta '(0,0.00469307)'
    AttrToken LPAR '('
    AttrTree low '0'
    AttrToken NUMERIC '0'
    AttrTree sep ','
    AttrToken COMMA ','
    AttrTree init '0.00469307'
    AttrToken NUMERIC '0.00469307'
    AttrToken RPAR ')'
    AttrTree ws ' '
    AttrToken WS_ALL ' '
    AttrTree comment '; CL'
    AttrToken SEMICOLON ';'
    AttrToken WS ' '
    AttrToken COMMENT 'CL'
    AttrTree ws '\n'
    AttrToken WS_ALL '\n'

.. _lark-parser: https://pypi.org/project/lark-parser/
