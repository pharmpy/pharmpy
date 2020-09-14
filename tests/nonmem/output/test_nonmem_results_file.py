from pathlib import Path

import pytest
from numpy import nan

import pharmpy.plugins.nonmem.results_file as rf
import pharmpy.plugins.nonmem.table as table


def test_data_io(pheno_lst):
    rfile = rf.NONMEMResultsFile(pheno_lst)
    assert rfile.nonmem_version == "7.4.2"


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [('phenocorr.lst', 1, {'minimization_successful': True,
                           'estimate_near_boundary': False,
                           'rounding_errors': False,
                           'maxevals_exceeded': False,
                           'significant_digits': 4.9,
                           'function_evaluations': 98,
                           'estimation_warning': False}, True),
     ('hessian_error.lst', 1, {'minimization_successful': False,
                               'estimate_near_boundary': None,
                               'rounding_errors': None,
                               'maxevals_exceeded': None,
                               'significant_digits': nan,
                               'function_evaluations': nan,
                               'estimation_warning': None}, False),
     ('large_s_matrix_cov_fail.lst', 1, {'minimization_successful': True,
                                         'estimate_near_boundary': False,
                                         'rounding_errors': False,
                                         'maxevals_exceeded': False,
                                         'significant_digits': 3.1,
                                         'function_evaluations': 62,
                                         'estimation_warning': True}, False),
     ('nm710_fail_negV.lst', 1, {'minimization_successful': None,
                                 'estimate_near_boundary': None,
                                 'rounding_errors': None,
                                 'maxevals_exceeded': None,
                                 'significant_digits': nan,
                                 'function_evaluations': nan,
                                 'estimation_warning': None}, None),
     ('sparse_matrix_with_msfi.lst', 1, {'minimization_successful': True,
                                         'estimate_near_boundary': False,
                                         'rounding_errors': False,
                                         'maxevals_exceeded': False,
                                         'significant_digits': 3.1,
                                         'function_evaluations': 112,
                                         'estimation_warning': True}, True),
     ('warfarin_ddmore.lst', 1, {'minimization_successful': True,
                                 'estimate_near_boundary': False,
                                 'rounding_errors': False,
                                 'maxevals_exceeded': False,
                                 'significant_digits': nan,
                                 'function_evaluations': nan,
                                 'estimation_warning': False}, False),
     ('mox_fail_nonp.lst', 1, {'minimization_successful': False,
                               'estimate_near_boundary': False,
                               'rounding_errors': False,
                               'maxevals_exceeded': False,
                               'significant_digits': nan,
                               'function_evaluations': 153,
                               'estimation_warning': False}, False),
     ('mox_nocov_nonp.lst', 1, {'minimization_successful': False,
                                'estimate_near_boundary': False,
                                'rounding_errors': False,
                                'maxevals_exceeded': False,
                                'significant_digits': nan,
                                'function_evaluations': 153,
                                'estimation_warning': False}, False),
     ('pheno_nonp.lst', 1, {'minimization_successful': True,
                            'estimate_near_boundary': False,
                            'rounding_errors': False,
                            'maxevals_exceeded': False,
                            'significant_digits': 3.6,
                            'function_evaluations': 107,
                            'estimation_warning': False}, True),
     ('theo.lst', 1, {'minimization_successful': True,
                      'estimate_near_boundary': False,
                      'rounding_errors': False,
                      'maxevals_exceeded': False,
                      'significant_digits': 4.2,
                      'function_evaluations': 208,
                      'estimation_warning': False}, True),
     ('theo_nonp.lst', 1, {'minimization_successful': False,
                           'estimate_near_boundary': True,
                           'rounding_errors': True,
                           'maxevals_exceeded': False,
                           'significant_digits': nan,
                           'function_evaluations': 735,
                           'estimation_warning': False}, False),
     ('theo_withcov.lst', 1, {'minimization_successful': True,
                              'estimate_near_boundary': False,
                              'rounding_errors': False,
                              'maxevals_exceeded': False,
                              'significant_digits': 4.2,
                              'function_evaluations': 208,
                              'estimation_warning': False}, True),
     ('UseCase7.lst', 1, {'minimization_successful': True,
                          'estimate_near_boundary': False,
                          'rounding_errors': False,
                          'maxevals_exceeded': False,
                          'significant_digits': nan,
                          'function_evaluations': nan,
                          'estimation_warning': False}, False),
     ('example6b_V7_30_beta.lst', 1, {'minimization_successful': True,
                                      'estimate_near_boundary': False,
                                      'rounding_errors': False,
                                      'maxevals_exceeded': False,
                                      'significant_digits': nan,
                                      'function_evaluations': nan,
                                      'estimation_warning': False}, False)
     ]
)
def test_estimation_status(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'noSIM')
    rfile = rf.NONMEMResultsFile(p / file)
    assert rfile.estimation_status(table_number) == expected
    if covariance_step_ok is None:
        assert rfile.covariance_status(table_number)['covariance_step_ok'] is None
    else:
        assert rfile.covariance_status(table_number)['covariance_step_ok'] == covariance_step_ok


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [('anneal2_V7_30_beta.lst', 2, {'minimization_successful': True,
                                    'estimate_near_boundary': False,
                                    'rounding_errors': False,
                                    'maxevals_exceeded': False,
                                    'significant_digits': nan,
                                    'function_evaluations': nan,
                                    'estimation_warning': False}, True),
     ('superid2_6_V7_30_beta.lst', 2, {'minimization_successful': True,
                                       'estimate_near_boundary': False,
                                       'rounding_errors': False,
                                       'maxevals_exceeded': False,
                                       'significant_digits': nan,
                                       'function_evaluations': nan,
                                       'estimation_warning': False}, True)
     ]
)
def test_estimation_status_multest(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM')
    rfile = rf.NONMEMResultsFile(p / file)
    assert rfile.estimation_status(table_number) == expected
    if covariance_step_ok is None:
        assert rfile.covariance_status(table_number)['covariance_step_ok'] is None
    else:
        assert rfile.covariance_status(table_number)['covariance_step_ok'] == covariance_step_ok


def test_estimation_status_withsim(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'control3boot.res')

    assert rfile.estimation_status(45) == {'minimization_successful': True,
                                           'estimate_near_boundary': False,
                                           'rounding_errors': False,
                                           'maxevals_exceeded': False,
                                           'significant_digits': 3.3,
                                           'function_evaluations': 192,
                                           'estimation_warning': False}
    assert rfile.covariance_status(45)['covariance_step_ok'] is False

    assert rfile.estimation_status(70) == {'minimization_successful': True,
                                           'estimate_near_boundary': True,
                                           'rounding_errors': False,
                                           'maxevals_exceeded': False,
                                           'significant_digits': 3.6,
                                           'function_evaluations': 202,
                                           'estimation_warning': False}
    assert rfile.covariance_status(70)['covariance_step_ok'] is False

    assert rfile.estimation_status(100) == {'minimization_successful': True,
                                            'estimate_near_boundary': False,
                                            'rounding_errors': False,
                                            'maxevals_exceeded': False,
                                            'significant_digits': 5.6,
                                            'function_evaluations': 100,
                                            'estimation_warning': False}
    assert rfile.covariance_status(100)['covariance_step_ok'] is True


def test_ofv_table_gap(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'multPROB' / 'multEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'multprobmix_nm730.lst')

    assert rfile.estimation_status(2) == {'minimization_successful': False,
                                          'estimate_near_boundary': False,
                                          'rounding_errors': False,
                                          'maxevals_exceeded': True,
                                          'significant_digits': nan,
                                          'function_evaluations': 16,
                                          'estimation_warning': False}

    table_numbers = (1, 2, 3, 4, 6, 8, 10, 11, 12, 13)
    ext_table_file = table.NONMEMTableFile(p / 'multprobmix_nm730.ext')

    for n in table_numbers:
        assert rfile.ofv(n) == pytest.approx(ext_table_file.table_no(n).final_ofv)
