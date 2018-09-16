
from pathlib import Path
from tempfile import TemporaryDirectory

from pharmpy import Model


def test_model_copy(pheno):
    """Copy test (same data in different objects)."""
    copy = pheno.copy()
    assert pheno is not copy
    assert pheno.path is not copy.path
    assert pheno.path.samefile(copy.path)
    for api in ['input', 'output', 'parameters', 'execute']:
        assert getattr(pheno, api) is not getattr(copy, api)


def test_model_path_set(pheno, pheno_path):
    """Change model filesystem path."""
    new_path = pheno_path.parent / 'will_not_exist.mod'

    # on copy
    copy = pheno.copy(str(new_path), write=False)
    assert not copy.exists and pheno.exists
    assert str(copy.path) == str(new_path)

    # manually
    copy.path = str(pheno_path)
    assert copy.exists
    assert str(copy.path) == str(pheno_path)


def test_model_write(pheno, pheno_path):
    """Test model write-on-copy."""
    tempdir = TemporaryDirectory()
    new_path = Path(tempdir.name) / ('%s_copy%s' % (pheno_path.stem, pheno_path.suffix))

    copy = pheno.copy(str(new_path), write=False)
    assert pheno.exists
    assert not copy.exists

    copy.write()
    assert pheno.exists
    assert copy.exists
    assert pheno.path.samefile(pheno_path)
    assert copy.path.samefile(new_path)

    new_path.unlink()
    assert not new_path.exists()

    copy = pheno.copy(str(new_path), write=True)
    assert pheno.exists
    assert copy.exists
    assert pheno.path.samefile(pheno_path)
    assert copy.path.samefile(new_path)


def test_model_de_novo():
    """Create model de novo, without existing file."""
    empty = Model()
    assert not empty.exists
    assert empty.path is None
    assert empty.content is None
