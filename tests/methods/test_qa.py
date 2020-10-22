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
