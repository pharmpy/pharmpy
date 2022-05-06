# def test_amd(tmp_path, testdata):
#     with TemporaryDirectoryChanger(tmp_path):
#         shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
#         dipath = Path(pharmpy.modeling.__file__).parent / 'example_models' / 'pheno.datainfo'
#         shutil.copy2(dipath, tmp_path)
#         res = run_amd(tmp_path / 'pheno.dta', mfl='LAGTIME();PERIPHERALS(1)')
#         assert res
