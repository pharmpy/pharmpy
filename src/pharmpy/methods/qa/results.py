from pathlib import Path

from pharmpy.results import Results


class QAResults(Results):
    def __init__(self):
        pass


def calculate_results():
    res = QAResults()
    return res


def psn_qa_results(path):
    """Create qa results from a PsN qa run

    :param path: Path to PsN qa run directory
    :return: A :class:`BootstrapResults` object
    """
    path = Path(path)

    res = calculate_results()
    return res
