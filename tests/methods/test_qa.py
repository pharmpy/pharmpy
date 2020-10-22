from io import StringIO

import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.methods.qa.results import calculate_results


def test_fullblock(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    fb = Model(testdata / 'nonmem' / 'qa' / 'fullblock.mod')
    res = calculate_results(orig, fullblock_model=fb)
    correct = """,new,old
"OMEGA(1,1)",0.486600,0.333246
"OMEGA(2,1)",0.846728,NaN
"OMEGA(2,2)",0.423262,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.fullblock_parameters, correct)

    assert res.dofv['dofv']['fullblock'] == pytest.approx(730.89472681373070 - 706.36113798726512)
    assert res.dofv['df']['fullblock'] == 1

    res = calculate_results(orig, fullblock_model=None)
    assert res.fullblock_parameters is None


def test_boxcox(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    bc = Model(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    res = calculate_results(orig, boxcox_model=bc)
    correct = """lambda,new_sd,old_sd
ETA(1),-1.581460,0.296257,0.333246
ETA(2),0.645817,0.429369,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.boxcox_parameters, correct)

    assert res.dofv['dofv']['boxcox'] == pytest.approx(730.89472681373070 - 721.78812733817688)
    assert res.dofv['df']['boxcox'] == 2

    res = calculate_results(orig, boxcox_model=None)
    assert res.boxcox_parameters is None


def test_tdist(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    td = Model(testdata / 'nonmem' / 'qa' / 'tdist.mod')
    res = calculate_results(orig, tdist_model=td)
    correct = """df,new_sd,old_sd
ETA(1),3.77,0.344951,0.333246
ETA(2),3.77,0.400863,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.tdist_parameters, correct)

    assert res.dofv['dofv']['tdist'] == pytest.approx(730.89472681373070 - 729.45800311609150)
    assert res.dofv['df']['tdist'] == 2

    res = calculate_results(orig, tdist_model=None)
    assert res.tdist_parameters is None
