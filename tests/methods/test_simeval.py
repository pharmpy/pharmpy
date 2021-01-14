from pharmpy.methods.simeval.results import psn_simeval_results


def test_psn_simeval_results(testdata):
    psn_simeval_results(testdata / 'psn' / 'simeval_dir1')
