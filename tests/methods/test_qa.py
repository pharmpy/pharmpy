from io import StringIO

import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.methods.qa.results import calculate_results
from pharmpy.results import read_results


def test_add_etas(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    add_etas = Model(testdata / 'nonmem' / 'qa' / 'add_etas_linbase.mod')
    res = calculate_results(orig, base, add_etas_model=add_etas, etas_added_to=['CL', 'V'])
    correct = """added,new_sd,orig_sd
ETA(1),True,0.338974,0.333246
ETA(2),True,0.449430,0.448917
CL,False,0.010001,NaN
V,False,0.010000,NaN
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.add_etas_parameters, correct, atol=1e-6)

    assert res.dofv['dofv']['parameter_variability', 'add_etas'] == pytest.approx(
        730.89472681373070 - 730.84697789365532
    )
    assert res.dofv['df']['parameter_variability', 'add_etas'] == 2


def test_fullblock(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    fb = Model(testdata / 'nonmem' / 'qa' / 'fullblock.mod')
    res = calculate_results(orig, base, fullblock_model=fb)
    correct = """,new,old
"OMEGA(1,1)",0.486600,0.333246
"OMEGA(2,1)",0.846728,NaN
"OMEGA(2,2)",0.423262,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.fullblock_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'fullblock'] == pytest.approx(
        730.89472681373070 - 706.36113798726512
    )
    assert res.dofv['df']['parameter_variability', 'fullblock'] == 1

    res = calculate_results(orig, base, fullblock_model=None)
    assert res.fullblock_parameters is None


def test_boxcox(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    bc = Model(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    res = calculate_results(orig, base, boxcox_model=bc)
    correct = """lambda,new_sd,old_sd
ETA(1),-1.581460,0.296257,0.333246
ETA(2),0.645817,0.429369,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.boxcox_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'boxcox'] == pytest.approx(
        730.89472681373070 - 721.78812733817688
    )
    assert res.dofv['df']['parameter_variability', 'boxcox'] == 2

    res = calculate_results(orig, base, boxcox_model=None)
    assert res.boxcox_parameters is None


def test_tdist(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    td = Model(testdata / 'nonmem' / 'qa' / 'tdist.mod')
    res = calculate_results(orig, base, tdist_model=td)
    correct = """df,new_sd,old_sd
ETA(1),3.77,0.344951,0.333246
ETA(2),3.77,0.400863,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.tdist_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'tdist'] == pytest.approx(
        730.89472681373070 - 729.45800311609150
    )
    assert res.dofv['df']['parameter_variability', 'tdist'] == 2

    res = calculate_results(orig, base, tdist_model=None)


def test_iov(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    iov = Model(testdata / 'nonmem' / 'qa' / 'iov.mod')
    res = calculate_results(orig, base, iov_model=iov)
    correct = """new_iiv_sd,orig_iiv_sd,iov_sd
ETA(1),0.259560,0.333246,0.555607
ETA(2),0.071481,0.448917,0.400451
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.iov_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'iov'] == pytest.approx(42.314986)
    assert res.dofv['df']['parameter_variability', 'iov'] == 2


def test_scm(testdata):
    orig = Model(testdata / 'nonmem' / 'pheno.mod')
    base = Model(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    scm_res = read_results(testdata / 'nonmem' / 'qa' / 'scm_results.json')
    res = calculate_results(orig, base, scm_results=scm_res)
    correct = """,,dofv,coeff
ETA(1),APGR,2.48792,-0.033334
ETA(1),WGT,0.48218,0.052342
ETA(2),APGR,0.59036,0.008371
ETA(2),WGT,0.00887,-0.003273
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    correct.index.set_names(['parameter', 'covariate'], inplace=True)
    pd.testing.assert_frame_equal(res.covariate_effects, correct, atol=1e-6)
    assert res.dofv['dofv']['covariates', 'ET1APGR-2'] == pytest.approx(2.48792)
    assert res.dofv['df']['covariates', 'ET1APGR-2'] == 1
