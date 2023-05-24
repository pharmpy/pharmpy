import pandas as pd
import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model.external.nonmem.table import CovTable, ExtTable, NONMEMTableFile, PhiTable


def test_nonmem_table(pheno_ext):
    ext_table_file = NONMEMTableFile(pheno_ext)
    ext_table = ext_table_file.table
    assert isinstance(ext_table, ExtTable)
    assert list(ext_table.data_frame.columns) == [
        'ITERATION',
        'THETA(1)',
        'THETA(2)',
        'THETA(3)',
        'OMEGA(1,1)',
        'OMEGA(2,1)',
        'OMEGA(2,2)',
        'SIGMA(1,1)',
        'OBJ',
    ]


def test_ext_table(pheno_ext):
    ext_table_file = NONMEMTableFile(pheno_ext)
    ext_table = ext_table_file.table
    assert isinstance(ext_table, ExtTable)
    assert ext_table.number == 1
    assert ext_table.is_evaluation is False
    assert ext_table.problem == 1
    assert ext_table.subproblem == 0
    assert ext_table.iteration2 == 0
    assert ext_table.method == "First Order Conditional Estimation with Interaction"
    assert ext_table.goal_function == "MINIMUM VALUE OF OBJECTIVE FUNCTION"
    assert ext_table.final_parameter_estimates['THETA(1)'] == 0.00469555
    assert list(ext_table.fixed) == [False, False, False, False, True, False, False]
    assert ext_table.final_ofv == 586.27605628188053
    assert ext_table.initial_ofv == 587.36644134661617


def test_phi_table(pheno_phi):
    phi_table_file = NONMEMTableFile(pheno_phi)
    phi_table = phi_table_file.table
    assert isinstance(phi_table, PhiTable)
    assert phi_table.problem == 1
    assert phi_table.subproblem == 0
    assert phi_table.iteration2 == 0
    assert phi_table.method == "First Order Conditional Estimation with Interaction"
    assert phi_table.goal_function is None


def test_cov_table(pheno_cov):
    cov_table_file = NONMEMTableFile(pheno_cov)
    cov_table = cov_table_file.table
    assert isinstance(cov_table, CovTable)
    df = cov_table.data_frame
    paramnames = ['THETA(1)', 'THETA(2)', 'THETA(3)', 'OMEGA(1,1)', 'OMEGA(2,2)', 'SIGMA(1,1)']
    assert list(df.columns) == paramnames
    assert list(df.index) == paramnames
    assert df.values[0][0] == 4.41151e-8
    assert df.values[0][3] == -1.09343e-6


def test_create_phi_table(tmp_path):
    with chdir(tmp_path):
        df = pd.DataFrame({'ETA(1)': [1, 2], 'ETA(2)': [5, 6]}, index=[1, 2])
        df.index.name = 'ID'
        phi = PhiTable(df=df)
        correct = (
            ' SUBJECT_NO   ID           ETA(1)       ETA(2)      \n'
            '            1            1  1.00000E+00  5.00000E+00\n'
            '            2            2  2.00000E+00  6.00000E+00\n'
        )
        assert phi.content == correct

        table_file = NONMEMTableFile(tables=[phi])
        table_file.write('my.phi')
        with open('my.phi', 'r') as fp:
            cont = fp.read()
        correct = 'TABLE NO.     1\n' + correct
        assert cont == correct


def test_errors(testdata):
    ext_file = NONMEMTableFile(testdata / 'nonmem' / 'errors' / 'no_header_error.ext')
    ext_table = ext_file.table_no(2)
    assert isinstance(ext_table, ExtTable)
    with pytest.raises(ValueError):
        ext_table.data_frame


def test_nonmemtablefile_notitle_github_issues_1251(tmp_path):
    filename = 'tab'
    with chdir(tmp_path):
        with open(filename, 'w') as fd:
            fd.write(
                ' ID          TIME        CWRES       CIPREDI     VC\n'
                '  1.1000E+02  0.0000E+00  0.0000E+00  0.0000E+00  9.5921E+01\n'
                '  1.1000E+02  0.0000E+00  0.0000E+00  0.0000E+00  9.5921E+01\n'
            )

        table_file = NONMEMTableFile(filename, notitle=True)
        table = table_file[0]
        df = table.data_frame

        assert tuple(df.columns) == ('ID', 'TIME', 'CWRES', 'CIPREDI', 'VC')
        assert len(df) == 2
