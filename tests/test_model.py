from pathlib import Path

import pytest

import pharmpy.data
import pharmpy.model
import pharmpy.symbols
from pharmpy import Model
from pharmpy.modeling import convert_model, create_symbol
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
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = create_symbol(model, stem=stem, force_numbering=force_numbering)

    assert symbol.name == symbol_name


def test_to_generic_model(testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    nm_model = Model.create_model(path)
    model = convert_model(nm_model, 'generic')

    assert id(model.parameters) != id(nm_model.parameters)
    assert model.parameters == nm_model.parameters
    assert id(model.random_variables) != id(nm_model.random_variables)
    assert model.random_variables == nm_model.random_variables
    assert model.name == nm_model.name
    assert id(model.statements) != id(nm_model.statements)
    assert model.statements == nm_model.statements
    assert type(model) == pharmpy.model.Model
