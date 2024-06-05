from pharmpy.tools import read_results
from pharmpy.tools.reporting import get_rst_path, report_available


def test_get_rst_path(testdata):
    res = read_results(testdata / 'results' / 'iivsearch_results.json')
    path = get_rst_path(res)
    assert path.name == 'report.rst'

    assert report_available(res)
