from pathlib import Path
from typing import Union

from pharmpy.internals.fs.path import normalize_user_given_path
from pharmpy.internals.fs.tmp import TemporaryDirectory
from pharmpy.workflows.results import Results


def create_report(results: Results, path: Union[Path, str]):
    """Create standard report for results

    The report will be an html created at specified path.

    Parameters
    ----------
    results : Results
        Results for which to create report
    path : Path
        Path to report file
    """
    import pharmpy.reporting.reporting as reporting

    path = normalize_user_given_path(path)

    with TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        json_path = tmp_path / 'results.json'
        results.to_json(json_path)
        reporting.generate_report(get_rst_path(results), json_path, path)


def get_rst_path(results):
    module = results.__module__
    a = module.split('.')
    if len(a) >= 3:
        tool = a[2]
    else:
        return None
    path = Path(__file__).parent / tool / 'report.rst'
    return path


def report_available(results):
    path = get_rst_path(results)
    return path is not None and path.is_file()
