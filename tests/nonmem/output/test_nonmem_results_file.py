from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from numpy import nan

import pharmpy.model.external.nonmem.table as table
import pharmpy.tools.external.nonmem.results_file as rf
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.workflows.log import Log

anan = pytest.approx(nan, nan_ok=True)


def _assert_estimation_status(_actual: rf.TermInfo, _expected: rf.TermInfo):
    expected = asdict(_expected)
    actual = asdict(_actual)

    assert actual.keys() == expected.keys()
    for key in expected.keys():
        assert type(actual[key]) is type(expected[key])
        if isinstance(expected[key], pd.DataFrame):
            assert str(actual[key]) == str(expected[key])
        elif expected[key] is nan:
            assert actual[key] is nan
        else:
            assert actual[key] == expected[key]


def test_supported_version():
    assert rf.NONMEMResultsFile.supported_version(None) is False
    assert rf.NONMEMResultsFile.supported_version('7.1.0') is False
    assert rf.NONMEMResultsFile.supported_version('7.2.0') is True
    assert rf.NONMEMResultsFile.supported_version('7.3.0') is True


def test_data_io(pheno_lst):
    rfile = rf.NONMEMResultsFile(pheno_lst)
    assert rfile.nonmem_version == "7.4.2"


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [
        (
            'phenocorr.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=4.9,
                function_evaluations=98,
                warning=False,
            ),
            True,
        ),
        (
            'hessian_error.lst',
            1,
            rf.TermInfo(
                minimization_successful=False,
            ),
            False,
        ),
        (
            'large_s_matrix_cov_fail.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=3.1,
                function_evaluations=62,
                warning=True,
                ebv_shrinkage=pd.DataFrame(
                    data={
                        "EBVshrink(%):": [
                            3.6501e01,
                            3.6538e01,
                            5.3119e01,
                            3.3266e01,
                            4.0104e01,
                            3.7548e01,
                            2.0563e01,
                            2.2176e01,
                            1.6328e01,
                            1.5847e01,
                            3.3517e01,
                            3.3526e01,
                            4.9502e01,
                            4.9855e01,
                        ]
                    },
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [2.3471e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={
                        "ETAshrink(%):": [
                            3.9737e01,
                            5.5124e01,
                            4.3564e01,
                            3.3676e01,
                            3.0325e01,
                            3.4879e01,
                            3.8156e01,
                            2.1724e01,
                            1.6271e01,
                            2.9704e01,
                            4.4234e01,
                            4.8502e01,
                            5.4318e01,
                            4.7864e01,
                        ]
                    },
                ),
            ),
            False,
        ),
        (
            'nm710_fail_negV.lst',
            1,
            rf.TermInfo(),
            None,
        ),
        (
            'sparse_matrix_with_msfi.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=3.1,
                function_evaluations=112,
                warning=True,
                ebv_shrinkage=pd.DataFrame(
                    data={
                        "EBVshrink(%):": [
                            4.5964e01,
                            3.4348e01,
                            5.2009e01,
                            3.3063e01,
                            3.3978e01,
                            3.7211e01,
                            2.1345e01,
                            2.3314e01,
                            1.5965e01,
                            1.6086e01,
                            3.2729e01,
                            3.3373e01,
                            5.1050e01,
                            4.9976e01,
                        ]
                    },
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [3.2079e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={
                        "ETAshrink(%):": [
                            4.7663e01,
                            5.2882e01,
                            4.3524e01,
                            3.6822e01,
                            3.8634e01,
                            3.5256e01,
                            4.0755e01,
                            2.8192e01,
                            2.0729e01,
                            2.9972e01,
                            4.9818e01,
                            4.8307e01,
                            6.6686e01,
                            4.9792e01,
                        ]
                    },
                ),
            ),
            True,
        ),
        (
            'warfarin_ddmore.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [7.6245e00]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [1.3927e00, 1.3092e01, 4.9181e01, 5.3072e01]},
                ),
            ),
            False,
        ),
        (
            'mox_fail_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful=False,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                function_evaluations=153,
                warning=False,
                ebv_shrinkage=pd.DataFrame(
                    data={"EBVshrink(%):": [3.7232e00, 2.5777e01, 1.4788e01, 2.4381e01, 1.6695e01]},
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [2.5697e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [3.1277e01, 5.2053e01, 7.5691e00, 7.8837e01, 8.2298e01]},
                ),
            ),
            False,
        ),
        (
            'mox_nocov_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful=False,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                function_evaluations=153,
                warning=False,
                ebv_shrinkage=pd.DataFrame(
                    data={"EBVshrink(%):": [3.7232e00, 2.5777e01, 1.4788e01, 2.4381e01, 1.6695e01]},
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [2.5697e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [3.1277e01, 5.2053e01, 7.5691e00, 7.8837e01, 8.2298e01]},
                ),
            ),
            False,
        ),
        (
            'pheno_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=3.6,
                function_evaluations=107,
                warning=False,
                ebv_shrinkage=pd.DataFrame(
                    data={"EBVshrink(%):": [3.8428e01, 4.4592e00]},
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [2.7971e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [3.8721e01, 4.6492e00]},
                ),
            ),
            True,
        ),
        (
            'theo.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=4.2,
                function_evaluations=208,
                warning=False,
            ),
            True,
        ),
        (
            'theo_nonp.lst',
            1,
            rf.TermInfo(
                minimization_successful=False,
                estimate_near_boundary=True,
                rounding_errors=True,
                maxevals_exceeded=False,
                function_evaluations=735,
                warning=False,
                ebv_shrinkage=pd.DataFrame(
                    data={"EBVshrink(%):": [9.6560e01, 1.6545e01, 1.6532e01, 1.0000e02]},
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [3.2692e00]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [9.6393e01, 1.2506e01, 1.2492e01, 1.0000e02]},
                ),
            ),
            False,
        ),
        (
            'theo_withcov.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                significant_digits=4.2,
                function_evaluations=208,
                warning=False,
            ),
            True,
        ),
        (
            'UseCase7.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                ebv_shrinkage=pd.DataFrame(
                    data={"EBVshrink(%):": [9.0057e00, 1.5538e01, 4.7502e01, 6.4241e01]},
                ),
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [1.3572e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [9.0067e00, 1.5526e01, 4.7506e01, 1.2182e01]},
                ),
            ),
            False,
        ),
        (
            'example6b_V7_30_beta.lst',
            1,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage=pd.DataFrame(
                    data={
                        "EPSshrink(%):": [1.5539e01, 6.8462e00],
                    },
                ),
                eta_shrinkage=pd.DataFrame(
                    data={
                        "ETAshrink(%):": [
                            6.3623e-01,
                            4.5490e00,
                            9.5378e00,
                            2.1243e00,
                            1.5371e00,
                            6.0355e00,
                            4.0732e-01,
                            1.7659e00,
                        ],
                    },
                ),
            ),
            False,
        ),
        (
            'maxeval3.lst',
            1,
            rf.TermInfo(
                minimization_successful=False,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=True,
                function_evaluations=5,
                warning=False,
                ofv_with_constant=3376.151276351326,
                ebv_shrinkage=pd.DataFrame(
                    data={
                        "EBVSHRINKSD(%)": [8.0993e00, 1.8003e00],
                        "EBVSHRINKVR(%)": [1.5543e01, 3.5683e00],
                    },
                ),
                eps_shrinkage=pd.DataFrame(
                    data={
                        "EPSSHRINKSD(%)": [1.0000e-10],
                        "EPSSHRINKVR(%)": [1.0000e-10],
                    },
                ),
                eta_shrinkage=pd.DataFrame(
                    data={
                        "ETASHRINKSD(%)": [1.0000e-10, 1.4424e01],
                        "ETASHRINKVR(%)": [1.0000e-10, 2.6768e01],
                    },
                ),
            ),
            False,
        ),
    ],
)
def test_estimation_status(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'noSIM')
    log = Log()
    rfile = rf.NONMEMResultsFile(p / file, log=log)
    actual = rfile.estimation_status(table_number)
    _assert_estimation_status(actual, expected)
    if covariance_step_ok is None:
        assert rfile.covariance_status(table_number).covariance_step_ok is None
    else:
        assert rfile.covariance_status(table_number).covariance_step_ok == covariance_step_ok


@pytest.mark.parametrize(
    'file, table_number, expected, covariance_step_ok',
    [
        (
            'anneal2_V7_30_beta.lst',
            2,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [-7.7375e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={"ETAshrink(%):": [4.7323e01, 3.9939e01, 2.4474e01, 0.0000e00]},
                ),
            ),
            True,
        ),
        (
            'superid2_6_V7_30_beta.lst',
            2,
            rf.TermInfo(
                minimization_successful=True,
                estimate_near_boundary=False,
                rounding_errors=False,
                maxevals_exceeded=False,
                warning=False,
                eps_shrinkage=pd.DataFrame(
                    data={"EPSshrink(%):": [1.1964e01]},
                ),
                eta_shrinkage=pd.DataFrame(
                    data={
                        "ETAshrink(%):": [
                            1.1115e01,
                            1.0764e01,
                            -1.9903e-01,
                            -4.3835e-02,
                            -9.1709e-02,
                            -1.9494e-02,
                        ]
                    },
                ),
            ),
            True,
        ),
    ],
)
def test_estimation_status_multest(testdata, file, table_number, expected, covariance_step_ok):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'multEST' / 'noSIM')
    rfile = rf.NONMEMResultsFile(p / file)
    _assert_estimation_status(rfile.estimation_status(table_number), expected)
    assert rfile.covariance_status(table_number).covariance_step_ok == covariance_step_ok


def test_estimation_status_empty():
    rfile = rf.NONMEMResultsFile()
    assert rfile._supported_nonmem_version is False
    assert rfile.estimation_status(1) == rf.TermInfo(
        significant_digits=anan,
        function_evaluations=anan,
    )


def test_estimation_status_withsim(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'onePROB' / 'oneEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'control3boot.res', log=Log())

    assert rfile.estimation_status(45) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=False,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=3.3,
        function_evaluations=192,
        warning=False,
    )
    assert rfile.covariance_status(45).covariance_step_ok is False

    assert rfile.estimation_status(70) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=True,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=3.6,
        function_evaluations=202,
        warning=False,
    )
    assert rfile.covariance_status(70).covariance_step_ok is False

    assert rfile.estimation_status(100) == rf.TermInfo(
        minimization_successful=True,
        estimate_near_boundary=False,
        rounding_errors=False,
        maxevals_exceeded=False,
        significant_digits=5.6,
        function_evaluations=100,
        warning=False,
    )
    assert rfile.covariance_status(100).covariance_step_ok is True


def test_ofv_table_gap(testdata):
    p = Path(testdata / 'nonmem' / 'modelfit_results' / 'multPROB' / 'multEST' / 'withSIM')
    rfile = rf.NONMEMResultsFile(p / 'multprobmix_nm730.lst', log=Log())

    _assert_estimation_status(
        rfile.estimation_status(2),
        rf.TermInfo(
            minimization_successful=False,
            estimate_near_boundary=False,
            rounding_errors=False,
            maxevals_exceeded=True,
            function_evaluations=16,
            warning=False,
            eta_shrinkage=pd.DataFrame(
                data={
                    'ETAshrink(%):': [1.7703, 12.038, 8.5112],
                },
            ),
            ebv_shrinkage=pd.DataFrame(
                data={
                    'EBVshrink(%):': [1.1841, 9.9088, 9.0686],
                },
            ),
            eps_shrinkage=pd.DataFrame(
                data={
                    'EPSshrink(%):': [10.166],
                },
            ),
        ),
    )

    table_numbers = (1, 2, 3, 4, 6, 8, 10, 11, 12, 13)
    ext_table_file = table.NONMEMTableFile(p / 'multprobmix_nm730.ext')

    for n in table_numbers:
        ext_table = ext_table_file.table_no(n)
        assert isinstance(ext_table, table.ExtTable)
        assert rfile.ofv(n) == pytest.approx(ext_table.final_ofv)


@pytest.mark.parametrize(
    'file_name, ref_start, no_of_rows, idx, no_of_errors',
    [
        (
            'control_stream_error.lst',
            'AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.',
            6,
            0,
            1,
        ),
        (
            'no_header_error.lst',
            'PRED EXIT CODE = 1',
            9,
            1,
            2,
        ),
        (
            'no_header_error.lst',
            'PROGRAM TERMINATED BY OBJ',
            2,
            2,
            2,
        ),
        (
            'rounding_error.lst',
            'MINIMIZATION TERMINATED\nDUE TO ROUNDING',
            2,
            0,
            2,
        ),
        (
            'zero_gradient_error.lst',
            'MINIMIZATION TERMINATED\nDUE TO ZERO',
            2,
            0,
            2,
        ),
        (
            'hessian.lst',
            'HESSIAN OF',
            1,
            0,
            1,
        ),
    ],
)
def test_errors(testdata, file_name, ref_start, no_of_rows, idx, no_of_errors):
    lst = rf.NONMEMResultsFile(testdata / 'nonmem' / 'errors' / file_name, log=Log())
    log_df = lst.log.to_dataframe()
    message = log_df['message'].iloc[idx]
    assert message.startswith(ref_start)
    assert len(message.split('\n')) == no_of_rows
    assert log_df['category'].value_counts()['ERROR'] == no_of_errors


@pytest.mark.parametrize(
    'file_name, ref, idx',
    [
        (
            'no_header_error.lst',
            'THE NUMBER OF PARAMETERS TO BE ESTIMATED\n'
            'EXCEEDS THE NUMBER OF INDIVIDUALS WITH DATA.',
            0,
        ),
        (
            'estimate_near_boundary_warning.lst',
            'PARAMETER ESTIMATE IS NEAR ITS BOUNDARY',
            0,
        ),
        (
            'est_step_warning.lst',
            'MINIMIZATION SUCCESSFUL\nHOWEVER, PROBLEMS OCCURRED WITH THE MINIMIZATION.',
            0,
        ),
    ],
)
def test_warnings(testdata, file_name, ref, idx):
    lst = rf.NONMEMResultsFile(testdata / 'nonmem' / 'errors' / file_name, log=Log())
    message = lst.log.to_dataframe()['message'].iloc[idx]
    assert message == ref


def test_covariance_status(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'modelfit_results' / 'covariance' / 'pheno_nocovariance.mod'
    model = load_model_for_test(path)
    res = parse_modelfit_results(model, path)
    assert all(res.standard_errors.isna())
    assert res.covariance_matrix is None
    assert res.correlation_matrix is None
    assert res.precision_matrix is None
