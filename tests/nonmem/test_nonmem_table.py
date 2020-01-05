import numpy as np
import pytest

import pharmpy.plugins.nonmem.table as table


def test_nonmem_table(pheno_ext):
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    assert list(ext_table_file.table.data_frame.columns) == ['ITERATION', 'THETA1', 'THETA2',
                                                             'THETA3', 'SIGMA(1,1)', 'OMEGA(1,1)',
                                                             'OMEGA(2,1)', 'OMEGA(2,2)', 'OBJ']


def test_ext_table(pheno_ext):
    ext_table_file = table.NONMEMTableFile(pheno_ext)
    ext_table = ext_table_file.table
    assert ext_table.problem == 1
    assert ext_table.subproblem == 0
    assert ext_table.iteration2 == 0
    assert ext_table.method == "First Order Conditional Estimation with Interaction"
    assert ext_table.goal_function == "MINIMUM VALUE OF OBJECTIVE FUNCTION"
    assert ext_table.final_parameter_estimates['THETA1'] == 0.00469555
    assert list(ext_table.fixed) == [False, False, False, False, False, True, False]
    assert ext_table.final_OFV == 586.27605628188053
    assert ext_table.initial_OFV == 587.36644134661617


def test_phi_table(pheno_phi):
    phi_table_file = table.NONMEMTableFile(pheno_phi)
    phi_table = phi_table_file.table
    assert phi_table.problem == 1
    assert phi_table.subproblem == 0
    assert phi_table.iteration2 == 0
    assert phi_table.method == "First Order Conditional Estimation with Interaction"
    assert phi_table.goal_function is None
    df = phi_table.data_frame
    assert df['ID'][0] == 1
    assert df['ID'][10] == 11
    assert df['ETA'][0] == [-4.38608E-02, 5.43031E-03]
    assert df['ETA'][10] == [6.61505E-02, 2.93685E-01]
    assert df['OBJ'][0] == pytest.approx(5.9473520242962552)
    assert df['OBJ'][10] == pytest.approx(9.4801638108538970)
    assert np.allclose(df['ETC'][0], np.array([[2.48833E-02, -2.99920E-03],
                                               [-2.99920E-03, 7.15713E-03]]))
    assert np.allclose(df['ETC'][10], np.array([[2.85322E-02, -3.63402E-03],
                                                [-3.63402E-03, 1.17722E-02]]))
