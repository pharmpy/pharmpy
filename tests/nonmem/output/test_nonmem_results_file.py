
def test_data_io(nonmem, pheno_lst):
    rfile = nonmem.output.NONMEMResultsFile(pheno_lst)
    print(rfile.nonmem_version)
#    data_io = nonmem.input.NMTRANDataIO(pheno_data, '@')
#    assert data_io.read(7) == '      1'
#    data_io = nonmem.input.NMTRANDataIO(pheno_data, 'I')
#    assert data_io.read(13) == '      1    0.'
#    data_io = nonmem.input.NMTRANDataIO(pheno_data, 'Q')
#    assert data_io.read(5) == 'ID TI'
