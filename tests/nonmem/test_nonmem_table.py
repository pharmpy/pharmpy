import pharmpy.plugins.nonmem.table as table


def test_nonmem_table(pheno_ext):
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    assert list(ext_table_file.table.data_frame.columns) == ['ITERATION', 'THETA(1)', 'THETA(2)',
                                                             'THETA(3)', 'OMEGA(1,1)',
                                                             'OMEGA(2,1)', 'OMEGA(2,2)',
                                                             'SIGMA(1,1)', 'OBJ']


def test_ext_table(pheno_ext):
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    ext_table = ext_table_file.table
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
