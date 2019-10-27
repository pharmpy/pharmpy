from pharmpy.plugins.nonmem.model import Model
from pharmpy.parameter import Parameter


def test_parameters(pheno_path):
    model = Model(pheno_path)
    params = model.parameters
    assert len(params) == 3     # FIXME: This is correct only for now.
    assert model.parameters['THETA(1)'] == Parameter('THETA(1)', 0.00469307, lower=0, upper=1000000)
    assert model.parameters['THETA(2)'] == Parameter('THETA(2)', 1.00916, lower=0, upper=1000000)
    assert model.parameters['THETA(3)'] == Parameter('THETA(3)', 0.1, lower=-0.99, upper=1000000)


#from pathlib import Path
#from tempfile import TemporaryDirectory

#from pharmpy import Model


#def test_model_copy(pheno):
#    """Copy test (same data in different objects)."""
#    copy = pheno.copy()
#    assert pheno is not copy
#    assert pheno.path is not copy.path
#    assert pheno.path.samefile(copy.path)
#    for api in ['input', 'output', 'parameters', 'execute']:
#        assert getattr(pheno, api) is not getattr(copy, api)


#def test_model_path_set(pheno, pheno_path):
#    """Change model filesystem path."""
#    new_path = pheno_path.parent / 'will_not_exist.mod'

    # on copy
#    copy = pheno.copy(str(new_path), write=False)
#    assert not copy.source.on_disk and pheno.source.on_disk
#    assert str(copy.path) == str(new_path)

    # manually
#    copy.path = str(pheno_path)
#    assert copy.source.on_disk
#    assert str(copy.path) == str(pheno_path)


#def test_model_write(pheno, pheno_path):
#    """Test model write-on-copy."""
#    tempdir = TemporaryDirectory()
#    new_path = Path(tempdir.name) / ('%s_copy%s' % (pheno_path.stem, pheno_path.suffix))

#    copy = pheno.copy(str(new_path), write=False)
#    assert pheno.source.on_disk
#    assert not copy.source.on_disk

#    copy.write()
#    assert pheno.source.on_disk
#    assert copy.source.on_disk
#    assert pheno.path.samefile(pheno_path)
#    assert copy.path.samefile(new_path)

#    new_path.unlink()
#    assert not new_path.exists()

#    copy = pheno.copy(str(new_path), write=True)
#    assert pheno.source.on_disk
#    assert copy.source.on_disk
#    assert pheno.path.samefile(pheno_path)
#    assert copy.path.samefile(new_path)


#def test_model_de_novo():
#    """Create model de novo, without existing file."""
#    empty = Model()
#    assert not empty.source.on_disk
#    assert empty.path is None
#    assert empty.content is None
