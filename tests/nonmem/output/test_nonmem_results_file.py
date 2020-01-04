import pharmpy.plugins.nonmem.results_file as rf


def test_data_io(pheno_lst):
    rfile = rf.NONMEMResultsFile(pheno_lst)
    assert rfile.nonmem_version == "7.4.2"
