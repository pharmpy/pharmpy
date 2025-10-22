from functools import partial
from io import StringIO

import pytest

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import NormalDistribution
from pharmpy.modeling import (
    add_iov,
    create_joint_distribution,
    create_rng,
    transform_etas_boxcox,
    transform_etas_tdist,
)
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.qa.results import calc_transformed_etas, calculate_results, psn_qa_results
from pharmpy.tools.qa.tool import (
    SECTIONS,
    categorize_model_entries,
    create_candidate,
    create_dofv_table,
    create_eta_transformation_results,
    validate_input,
)
from pharmpy.tools.ruvsearch.results import psn_resmod_results
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.results import read_results


@pytest.mark.parametrize(
    'trans_type, added_etas',
    [('tdist', 3), ('boxcox', 3), ('fullblock', 3)],
)
def test_create_candidate(load_model_for_test, testdata, trans_type, added_etas):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_base = load_model_for_test(path)
    res_base = parse_modelfit_results(model_base, path)
    me_base = ModelEntry.create(model_base, modelfit_results=res_base)

    me = create_candidate(me_base, trans_type)
    assert me.model.name == trans_type
    assert me.modelfit_results is None
    assert me.parent == model_base

    assert len(me.model.parameters) - len(model_base.parameters) == added_etas


def test_categorize_model_entries(load_model_for_test, testdata, model_entry_factory):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_base = load_model_for_test(path)
    res_base = parse_modelfit_results(model_base, path)
    me_base = ModelEntry(model_base, modelfit_results=res_base)

    model1 = transform_etas_tdist(model_base).replace(name='tdist')
    model2 = transform_etas_boxcox(model_base).replace(name='boxcox')
    model3 = create_joint_distribution(model_base).replace(name='fullblock')

    me_cands = model_entry_factory(
        [model1, model2, model3], ref_val=res_base.ofv, parent=model_base
    )

    base, cands = categorize_model_entries([me_base] + me_cands)
    assert base.model.name == me_base.model.name
    assert len(cands) == len(me_cands)
    assert base not in cands

    with pytest.raises(ValueError):
        me_invalid = ModelEntry.create(model_base.replace(name='x'))
        categorize_model_entries([me_base] + me_cands + [me_invalid])


def test_create_dofv_table(load_model_for_test, testdata, model_entry_factory):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_base = load_model_for_test(path)
    res_base = parse_modelfit_results(model_base, path)
    me_base = ModelEntry(model_base, modelfit_results=res_base)

    model1 = transform_etas_tdist(model_base).replace(name='tdist')
    model2 = transform_etas_boxcox(model_base).replace(name='boxcox')
    model3 = create_joint_distribution(model_base).replace(name='fullblock')

    me_cands = model_entry_factory(
        [model1, model2, model3], ref_val=res_base.ofv, parent=model_base
    )

    table = create_dofv_table(me_base, me_cands)

    assert len(table) == len(me_cands)
    for me_base in me_cands:
        assert table.loc[me_base.model.name]['dofv'] == res_base.ofv - me_base.modelfit_results.ofv
        assert table.loc[me_base.model.name]['added_params'] == len(me_base.model.parameters) - len(
            model_base.parameters
        )


def test_create_eta_transformation_results(load_model_for_test, testdata, model_entry_factory):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model_base = load_model_for_test(path)
    res_base = parse_modelfit_results(model_base, path)
    me_base = ModelEntry(model_base, modelfit_results=res_base)

    model1 = transform_etas_tdist(model_base).replace(name='tdist')
    model2 = transform_etas_boxcox(model_base).replace(name='boxcox')

    me_cands = model_entry_factory([model1, model2], ref_val=res_base.ofv, parent=model_base)

    for me in me_cands:
        rng = create_rng(1)
        res_dict = create_eta_transformation_results(me, me_base, rng)
        table = res_dict[f'{me.model.name}_parameters']
        param_name = 'lambda' if me.model.name == 'boxcox' else 'df'
        assert table[param_name].is_unique
        assert (table['old_sd'] != table['new_sd']).all()
        plot = res_dict[f'{me.model.name}_plot']
        assert plot


def test_add_etas(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    add_etas = load_model_for_test(testdata / 'nonmem' / 'qa' / 'add_etas_linbase.mod')
    add_etas_res = parse_modelfit_results(
        add_etas, testdata / 'nonmem' / 'qa' / 'add_etas_linbase.mod'
    )
    add_etas_entry = ModelEntry.create(model=add_etas, modelfit_results=add_etas_res)
    res = calculate_results(
        orig_entry, base_entry, add_etas_model_entry=add_etas_entry, etas_added_to=['CL', 'V']
    )
    correct = """added,new_sd,orig_sd
ETA_1,True,0.338974,0.333246
ETA_2,True,0.449430,0.448917
CL,False,0.010001,NaN
V,False,0.010000,NaN
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.add_etas_parameters, correct, atol=1e-6)

    assert res.dofv['dofv']['parameter_variability', 'add_etas', np.nan] == pytest.approx(
        730.89472681373070 - 730.84697789365532
    )
    assert res.dofv['df']['parameter_variability', 'add_etas', np.nan] == 2


def test_fullblock(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    fb = load_model_for_test(testdata / 'nonmem' / 'qa' / 'fullblock.mod')
    fb_res = parse_modelfit_results(fb, testdata / 'nonmem' / 'qa' / 'fullblock.mod')
    fb_entry = ModelEntry.create(model=fb, modelfit_results=fb_res)
    res = calculate_results(orig_entry, base_entry, fullblock_model_entry=fb_entry)
    correct = """,new,old
"IVCL",0.486600,0.333246
"OMEGA_2_1",0.846728,NaN
"IVV",0.423262,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.fullblock_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'fullblock', np.nan] == pytest.approx(
        730.89472681373070 - 706.36113798726512
    )
    assert res.dofv['df']['parameter_variability', 'fullblock', np.nan] == 1

    res = calculate_results(orig_entry, base_entry, fullblock_model_entry=None)
    assert res.fullblock_parameters is None


def test_boxcox(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    bc = load_model_for_test(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    bc_res = parse_modelfit_results(bc, testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    bc_entry = ModelEntry.create(model=bc, modelfit_results=bc_res)
    res = calculate_results(orig_entry, base_entry, boxcox_model_entry=bc_entry)
    correct = """lambda,new_sd,old_sd
ETA_1,-1.581460,0.296257,0.333246
ETA_2,0.645817,0.429369,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.boxcox_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'boxcox', np.nan] == pytest.approx(
        730.89472681373070 - 721.78812733817688
    )
    assert res.dofv['df']['parameter_variability', 'boxcox', np.nan] == 2

    res = calculate_results(orig_entry, base_entry, boxcox_model_entry=None)
    assert res.boxcox_parameters is None


def test_tdist(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    td = load_model_for_test(testdata / 'nonmem' / 'qa' / 'tdist.mod')
    td_res = parse_modelfit_results(td, testdata / 'nonmem' / 'qa' / 'tdist.mod')
    td_entry = ModelEntry.create(model=td, modelfit_results=td_res)
    res = calculate_results(orig_entry, base_entry, tdist_model_entry=td_entry)
    correct = """df,new_sd,old_sd
ETA_1,3.77,0.344951,0.333246
ETA_2,3.77,0.400863,0.448917
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.tdist_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'tdist', np.nan] == pytest.approx(
        730.89472681373070 - 729.45800311609150
    )
    assert res.dofv['df']['parameter_variability', 'tdist', np.nan] == 2

    res = calculate_results(orig_entry, base_entry, tdist_model_entry=None)


@pytest.mark.parametrize(
    'funcs_base, funcs_candidate, linearize, trans_type',
    [
        ([], [transform_etas_tdist], False, 'tdist'),
        ([], [transform_etas_boxcox], False, 'boxcox'),
        ([], [create_joint_distribution, transform_etas_tdist], False, 'tdist'),
        ([], [create_joint_distribution, transform_etas_boxcox], False, 'boxcox'),
        (
            [partial(add_iov, occ='VISI', list_of_parameters=['CL'])],
            [transform_etas_boxcox],
            False,
            'boxcox',
        ),
    ],
)
def test_calc_transformed_etas(
    load_model_for_test,
    testdata,
    model_entry_factory,
    funcs_base,
    funcs_candidate,
    linearize,
    trans_type,
):
    model_base = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs_base:
        model_base = func(model_base)
    model = model_base
    for func in funcs_candidate:
        model = func(model)

    mes = model_entry_factory([model_base, model])
    me_base, me = mes[0], mes[1]

    param_type = 'lambda' if trans_type == 'boxcox' else 'df'
    table, _ = calc_transformed_etas(me_base, me, trans_type, param_type)

    assert param_type in table.columns
    for i, idx in enumerate(table.index):
        sd_new = table.loc[idx]['new_sd']
        sd_old = table.loc[idx]['old_sd']
        assert sd_new != sd_old
        if idx in model.random_variables.iov.names:
            eta = model.random_variables[idx]
            assert isinstance(eta, NormalDistribution)
            omega_est = me.modelfit_results.parameter_estimates[eta.variance.name]
        else:
            omega = model.random_variables.variance_parameters[i]
            omega_est = me.modelfit_results.parameter_estimates[omega]
        assert omega_est**0.5 == sd_new


def test_iov(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    iov = load_model_for_test(testdata / 'nonmem' / 'qa' / 'iov.mod')
    iov_res = parse_modelfit_results(iov, testdata / 'nonmem' / 'qa' / 'iov.mod')
    iov_entry = ModelEntry.create(model=iov, modelfit_results=iov_res)
    res = calculate_results(orig_entry, base_entry, iov_model_entry=iov_entry)
    correct = """new_iiv_sd,orig_iiv_sd,iov_sd
ETA_1,0.259560,0.333246,0.555607
ETA_2,0.071481,0.448917,0.400451
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0])
    pd.testing.assert_frame_equal(res.iov_parameters, correct)

    assert res.dofv['dofv']['parameter_variability', 'iov', np.nan] == pytest.approx(42.314986)
    assert res.dofv['df']['parameter_variability', 'iov', np.nan] == 2


def test_scm(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    scm_res = read_results(testdata / 'nonmem' / 'qa' / 'scm_results.json')
    res = calculate_results(orig_entry, base_entry, scm_results=scm_res)
    correct = """,,dofv,coeff
ETA(1),APGR,2.48792,-0.033334
ETA(1),WGT,0.48218,0.052342
ETA(2),APGR,0.59036,0.008371
ETA(2),WGT,0.00887,-0.003273
"""
    correct = pd.read_csv(StringIO(correct), index_col=[0, 1])
    correct.index.set_names(['parameter', 'covariate'], inplace=True)
    pd.testing.assert_frame_equal(res.covariate_effects, correct, atol=1e-6)
    assert res.dofv['dofv']['covariates', 'ET1APGR-2', np.nan] == pytest.approx(2.48792)
    assert res.dofv['df']['covariates', 'ET1APGR-2', np.nan] == 1


def test_resmod(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    resmod_res = read_results(testdata / 'nonmem' / 'qa' / 'resmod_results.json')
    res = calculate_results(orig_entry, base_entry, resmod_idv_results=resmod_res)
    assert list(res.residual_error['additional_parameters']) == [2, 2, 1, 1, 1, 1]
    assert list(res.residual_error['dOFV']) == [13.91, 8.03, 5.53, 3.34, 1.31, 0.03]
    assert res.residual_error.index.tolist() == [
        (1, 'dtbs'),
        (1, 'time_varying'),
        (1, 'tdist'),
        (1, 'autocorrelation'),
        (1, 'IIV_on_RUV'),
        (1, 'power'),
    ]
    assert res.dofv['dofv']['residual_error_model', 'dtbs', 1] == pytest.approx(13.91)


def test_resmod_dvid(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_res = parse_modelfit_results(orig, testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig, modelfit_results=orig_res)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_res = parse_modelfit_results(base, testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base, modelfit_results=base_res)
    resmod_res = psn_resmod_results(testdata / 'psn' / 'resmod_dir2')
    res = calculate_results(orig_entry, base_entry, resmod_idv_results=resmod_res)
    assert res.residual_error.loc[("4", "tdist"), 'dOFV'] == 2.41


def test_psn_qa_results(testdata):
    path = testdata / 'psn' / 'qa_dir1'
    res = psn_qa_results(path)
    correct = """idv,dvid,binid,binmin,binmax,cwres,cpred
TIME,1,1,0.00,2.00,0.28,-6
TIME,1,2,2.00,2.55,0.12,-2
TIME,1,3,2.55,11.00,-0.29,6
TIME,1,4,11.00,47.25,0.04,-1
TIME,1,5,47.25,63.50,-0.39,7
TIME,1,6,63.50,83.10,0.20,-4
TIME,1,7,83.10,112.30,0.11,-2
TIME,1,8,112.30,135.50,-0.29,5
TIME,1,9,135.50,159.80,0.19,-4
TIME,1,10,159.80,390.00,-0.02,1
TAD,1,1,0.00,1.50,0.06,-1
TAD,1,2,1.50,2.00,0.41,-8
TAD,1,3,2.00,3.00,-0.13,3
TAD,1,4,3.00,6.00,-0.06,1
TAD,1,5,6.00,11.00,-0.18,3
TAD,1,6,11.00,11.50,0.54,-10
TAD,1,7,11.50,11.70,-0.24,4
TAD,1,8,11.70,14.00,0.06,-1
TAD,1,9,14.00,258.00,0.07,-1
PRED,1,1,8.00,17.67,0.19,-5
PRED,1,2,17.67,19.50,0.10,-1
PRED,1,3,19.50,20.13,-0.17,3
PRED,1,4,20.13,21.39,-0.01,0
PRED,1,5,21.39,24.32,0.24,-4
PRED,1,6,24.32,26.63,0.06,-1
PRED,1,7,26.63,28.70,0.05,-1
PRED,1,8,28.70,31.28,-0.05,1
PRED,1,9,31.28,36.34,0.07,0
PRED,1,10,36.34,54.00,-0.47,9
"""

    correct = pd.read_csv(StringIO(correct), index_col=[0, 1, 2])
    pd.testing.assert_frame_equal(res.structural_bias, correct, atol=1e-6)


def test_simeval(load_model_for_test, testdata):
    orig = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    orig_entry = ModelEntry.create(model=orig)
    base = load_model_for_test(testdata / 'nonmem' / 'qa' / 'pheno_linbase.mod')
    base_entry = ModelEntry.create(model=base)
    simeval_res = read_results(testdata / 'nonmem' / 'qa' / 'simeval_results.json')
    cdd_res = read_results(testdata / 'nonmem' / 'qa' / 'cdd_results.json')
    calculate_results(orig_entry, base_entry, simeval_results=simeval_res, cdd_results=cdd_res)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        (
            {'skip': list(SECTIONS)},
            'Invalid `skip`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    kwargs,
    match,
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = parse_modelfit_results(model, testdata / 'nonmem' / 'models' / 'mox2.mod')

    with pytest.raises(ValueError, match=match):
        validate_input(model, res, **kwargs)
