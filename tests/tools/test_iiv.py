from io import StringIO

import numpy as np
import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, add_peripheral_compartment, create_joint_distribution
from pharmpy.tools.iiv.algorithms import (
    _get_possible_iiv_blocks,
    _is_current_block_structure,
    brute_force_block_structure,
)
from pharmpy.tools.iiv.tool import _update_inits_start_model


@pytest.mark.parametrize(
    'list_of_parameters, block_structure, no_of_models',
    [([], [], 4), (['QP1'], [], 14), ([], ['ETA(1)', 'ETA(2)'], 4)],
)
def test_brute_force_block_structure(testdata, list_of_parameters, block_structure, no_of_models):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    add_peripheral_compartment(model)
    add_iiv(model, list_of_parameters, 'add')
    if block_structure:
        create_joint_distribution(model, block_structure)

    iivs = model.random_variables.iiv
    wf, model_features = brute_force_block_structure(iivs)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models
    assert block_structure not in model_features.items()


def test_get_iiv_combinations_4_etas(testdata, pheno_path):
    model = Model.create_model(
        StringIO(
            '''
$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V*EXP(ETA(3))*EXP(ETA(4))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA DIAGONAL(4)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
 0.031128
 0.031128
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
        )
    )
    model.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset

    iivs = model.random_variables.iiv
    iiv_single_block, iiv_multi_block = _get_possible_iiv_blocks(iivs)

    assert iiv_single_block == [
        ['ETA(1)', 'ETA(2)'],
        ['ETA(1)', 'ETA(3)'],
        ['ETA(1)', 'ETA(4)'],
        ['ETA(2)', 'ETA(3)'],
        ['ETA(2)', 'ETA(4)'],
        ['ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)'],
        ['ETA(1)', 'ETA(2)', 'ETA(4)'],
        ['ETA(1)', 'ETA(3)', 'ETA(4)'],
        ['ETA(2)', 'ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)', 'ETA(4)'],
    ]

    assert iiv_multi_block == [
        [['ETA(1)', 'ETA(2)'], ['ETA(3)', 'ETA(4)']],
        [['ETA(1)', 'ETA(3)'], ['ETA(2)', 'ETA(4)']],
        [['ETA(1)', 'ETA(4)'], ['ETA(2)', 'ETA(3)']],
    ]


def test_get_iiv_combinations_5_etas(testdata, pheno_path):
    model = Model.create_model(
        StringIO(
            '''
$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V*EXP(ETA(3))*EXP(ETA(4))*EXP(ETA(5))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA DIAGONAL(5)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
 0.031128
 0.031128
 0.031128
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
        )
    )

    model.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset

    iivs = model.random_variables.iiv
    iiv_single_block, iiv_multi_block = _get_possible_iiv_blocks(iivs)

    assert iiv_single_block == [
        ['ETA(1)', 'ETA(2)'],
        ['ETA(1)', 'ETA(3)'],
        ['ETA(1)', 'ETA(4)'],
        ['ETA(1)', 'ETA(5)'],
        ['ETA(2)', 'ETA(3)'],
        ['ETA(2)', 'ETA(4)'],
        ['ETA(2)', 'ETA(5)'],
        ['ETA(3)', 'ETA(4)'],
        ['ETA(3)', 'ETA(5)'],
        ['ETA(4)', 'ETA(5)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)'],
        ['ETA(1)', 'ETA(2)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(5)'],
        ['ETA(1)', 'ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(3)', 'ETA(5)'],
        ['ETA(1)', 'ETA(4)', 'ETA(5)'],
        ['ETA(2)', 'ETA(3)', 'ETA(4)'],
        ['ETA(2)', 'ETA(3)', 'ETA(5)'],
        ['ETA(2)', 'ETA(4)', 'ETA(5)'],
        ['ETA(3)', 'ETA(4)', 'ETA(5)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)', 'ETA(4)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)', 'ETA(5)'],
        ['ETA(1)', 'ETA(2)', 'ETA(4)', 'ETA(5)'],
        ['ETA(1)', 'ETA(3)', 'ETA(4)', 'ETA(5)'],
        ['ETA(2)', 'ETA(3)', 'ETA(4)', 'ETA(5)'],
        ['ETA(1)', 'ETA(2)', 'ETA(3)', 'ETA(4)', 'ETA(5)'],
    ]

    assert iiv_multi_block == [
        [['ETA(1)', 'ETA(2)'], ['ETA(3)', 'ETA(4)', 'ETA(5)']],
        [['ETA(1)', 'ETA(3)'], ['ETA(2)', 'ETA(4)', 'ETA(5)']],
        [['ETA(1)', 'ETA(4)'], ['ETA(2)', 'ETA(3)', 'ETA(5)']],
        [['ETA(1)', 'ETA(5)'], ['ETA(2)', 'ETA(3)', 'ETA(4)']],
        [['ETA(2)', 'ETA(3)'], ['ETA(1)', 'ETA(4)', 'ETA(5)']],
        [['ETA(2)', 'ETA(4)'], ['ETA(1)', 'ETA(3)', 'ETA(5)']],
        [['ETA(2)', 'ETA(5)'], ['ETA(1)', 'ETA(3)', 'ETA(4)']],
        [['ETA(3)', 'ETA(4)'], ['ETA(1)', 'ETA(2)', 'ETA(5)']],
        [['ETA(3)', 'ETA(5)'], ['ETA(1)', 'ETA(2)', 'ETA(4)']],
        [['ETA(4)', 'ETA(5)'], ['ETA(1)', 'ETA(2)', 'ETA(3)']],
    ]


def test_is_current_block_structure(testdata):
    model_code = StringIO(
        '''
$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V*EXP(ETA(3))*EXP(ETA(4))

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA DIAGONAL(4)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
 0.031128
 0.031128
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
'''
    )
    model = Model.create_model(model_code)
    model.path = testdata / 'nonmem' / 'pheno.mod'  # To be able to find dataset
    list_of_etas_12 = ['ETA(1)', 'ETA(2)']
    create_joint_distribution(model, list_of_etas_12)
    assert _is_current_block_structure(model.random_variables.iiv, list_of_etas_12)
    list_of_etas_34 = ['ETA(3)', 'ETA(4)']
    assert not _is_current_block_structure(model.random_variables.iiv, list_of_etas_34)
    list_of_etas_23 = ['ETA(2)', 'ETA(3)']
    assert not _is_current_block_structure(model.random_variables.iiv, list_of_etas_23)
    create_joint_distribution(model)
    assert _is_current_block_structure(
        model.random_variables.iiv, list_of_etas_12 + list_of_etas_34
    )


def test_update_inits_start_model(pheno_path):
    model = Model.create_model(pheno_path)
    model.modelfit_results.parameter_estimates['OMEGA(1,1)'] = np.nan
    _update_inits_start_model(model)
