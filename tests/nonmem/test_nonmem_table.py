import pandas as pd

def test_nonmem_table(nonmem, pheno_ext):
    ext_table = nonmem.output.NONMEMTable(pheno_ext)
    assert list(ext_table.data_frame.columns) == ['ITERATION', 'THETA1', 'THETA2', 'THETA3', 'SIGMA(1,1)', 'OMEGA(1,1)',
         'OMEGA(2,1)', 'OMEGA(2,2)', 'OBJ']

def test_ext_table(nonmem, pheno_ext):
    ext_table = nonmem.output.ExtTable(pheno_ext)
    assert ext_table.final_parameter_estimates['THETA1'] == 0.00469555
    assert list(ext_table.fixed) == [ False, False, False, False, False, True, False ]
    assert ext_table.final_OFV == 586.27605628188053
    assert ext_table.initial_OFV == 587.36644134661617
