def test_data_io(nonmem, pheno_ext):
    ext_table = nonmem.output.NONMEMTable(pheno_ext)
    print(ext_table.data_frame)
