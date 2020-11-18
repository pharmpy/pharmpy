from pathlib import Path

from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.modeling import (
    fix_parameters,
    read_model,
    unfix_parameters,
    update_source,
    write_model,
)


def test_read_model(testdata):
    model = read_model(testdata / 'nonmem' / 'minimal.mod')
    assert model.parameters['THETA(1)'].init == 0.1
    model2 = read_model(str(testdata / 'nonmem' / 'minimal.mod'))
    assert model2.parameters['THETA(1)'].init == 0.1


def test_write_model(testdata):
    model = read_model(testdata / 'nonmem' / 'minimal.mod')
    with Patcher(additional_skip_names=['pkgutil']):
        write_model(model, 'run1.mod')
        assert Path('run1.mod').is_file()


def test_fix_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    assert not model.parameters['THETA(1)'].fix
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix


def test_unfix_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, ['THETA(1)'])
    assert not model.parameters['THETA(1)'].fix

    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, 'THETA(1)')
    assert model.parameters['THETA(1)'].fix
    unfix_parameters(model, 'THETA(1)')
    assert not model.parameters['THETA(1)'].fix


def test_update_source(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    fix_parameters(model, ['THETA(1)'])
    update_source(model)
    assert str(model).split('\n')[7] == '$THETA 0.1 FIX'
