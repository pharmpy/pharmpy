def create_report(results, path):
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
