from pathlib import Path
from typing import Union

from pharmpy.results import Results


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

    reporting.generate_report(results.rst_path, path)
