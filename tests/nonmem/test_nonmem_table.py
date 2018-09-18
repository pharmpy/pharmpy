import pandas as pd

def test_data_io(nonmem, pheno_ext):
    ext_table = nonmem.output.NONMEMTable(pheno_ext)
    assert list(ext_table.data_frame.columns) == ['ITERATION', 'THETA1', 'THETA2', 'THETA3', 'SIGMA(1,1)', 'OMEGA(1,1)',
         'OMEGA(2,1)', 'OMEGA(2,2)', 'OBJ']
