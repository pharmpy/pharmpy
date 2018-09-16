
from pharmpy import Model


def test_read(pheno_real):
    """Read test."""
    pheno = Model(str(pheno_real))
    assert pheno.path.samefile(pheno_real)
    with open(str(pheno_real), 'r') as f:
        buf = f.read()
    assert str(pheno) == pheno.content == buf


def test_model_copy(pheno_real):
    """Copy test (same data in different objects)."""
    pheno = Model(pheno_real)
    copy = pheno.copy()
    assert pheno is not copy
    assert pheno.path is not copy.path
    assert pheno.path.samefile(copy.path)
    for api in ['input', 'output', 'parameters', 'execute']:
        assert getattr(pheno, api) is not getattr(copy, api)


def test_model_path_set(pheno_real):
    """Change model filesystem path."""
    pheno = Model(pheno_real)
    new_path = pheno_real.parent / ('%s_copy%s' % (pheno_real.stem, pheno_real.suffix))

    # new on copy
    copy = pheno.copy(str(new_path))
    # assert copy.exists
    # assert copy.path.samefile(new_path)
    assert str(copy.path) == str(new_path)

    # manually
    pheno.path = str(new_path)
    assert not pheno.exists
    assert str(pheno.path) == str(new_path)


def test_model_de_novo():
    """Create model de novo, without existing file."""
    none = Model()
    assert none.path is None
    assert not none.exists
    assert none.content is None
