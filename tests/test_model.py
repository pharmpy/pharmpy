from pathlib import Path

import pytest

import pharmpy.data
import pharmpy.model
import pharmpy.symbols
from pharmpy import Model
from pharmpy.plugins.nonmem.dataset import read_nonmem_dataset

tabpath = Path(__file__).parent / 'testdata' / 'nonmem' / 'pheno_real_linbase.tab'
lincorrect = read_nonmem_dataset(
    tabpath,
    ignore_character='@',
    colnames=['ID', 'G11', 'G21', 'H11', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES'],
)


@pytest.mark.parametrize(
    'stem,force_numbering,symbol_name', [('DV', False, 'DV1'), ('X', False, 'X'), ('X', True, 'X1')]
)
def test_create_symbol(testdata, stem, force_numbering, symbol_name):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = model.create_symbol(stem=stem, force_numbering=force_numbering)

    assert symbol.name == symbol_name


def test_to_generic_model(testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    nm_model = Model(path)
    model = nm_model.to_generic_model()
    assert id(model.parameters) != id(nm_model.parameters)
    assert model.parameters == nm_model.parameters
    assert id(model.random_variables) != id(nm_model.random_variables)
    assert model.random_variables == nm_model.random_variables
    assert model.name == nm_model.name
    assert id(model.statements) != id(nm_model.statements)
    assert model.statements == nm_model.statements
    assert type(model) == pharmpy.model.Model
