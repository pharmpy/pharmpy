import pandas as pd

import pharmpy.plugins.nonmem.table as table


def test_nonmem_table(pheno_ext):
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    assert list(ext_table_file.table.data_frame.columns) == [
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
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    ext_table = ext_table_file.table
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
    phi_table_file = table.NONMEMTableFile(pheno_phi)
    phi_table = phi_table_file.table
    assert phi_table.problem == 1
    assert phi_table.subproblem == 0
    assert phi_table.iteration2 == 0
    assert phi_table.method == "First Order Conditional Estimation with Interaction"
    assert phi_table.goal_function is None


def test_cov_table(pheno_cov):
    cov_table_file = table.NONMEMTableFile(pheno_cov)
    cov_table = cov_table_file.table
    df = cov_table.data_frame
    paramnames = ['THETA(1)', 'THETA(2)', 'THETA(3)', 'OMEGA(1,1)', 'OMEGA(2,2)', 'SIGMA(1,1)']
    assert list(df.columns) == paramnames
    assert list(df.index) == paramnames
    assert df.values[0][0] == 4.41151e-8
    assert df.values[0][3] == -1.09343e-6


def test_create_phi_table(fs):
    df = pd.DataFrame({'ETA(1)': [1, 2], 'ETA(2)': [5, 6]}, index=[1, 2])
    df.index.name = 'ID'
    phi = table.PhiTable(df=df)
    phi.create_content()
    correct = ''' SUBJECT_NO   ID           ETA(1)       ETA(2)      
            1            1  1.00000E+00  5.00000E+00
            2            2  2.00000E+00  6.00000E+00
'''  # noqa W291
    assert phi.content == correct

    table_file = table.NONMEMTableFile(tables=[phi])
    table_file.write('my.phi')
    with open('my.phi', 'r') as fp:
        cont = fp.read()
    correct = 'TABLE NO.     1\n' + correct
    assert cont == correct
