from typing import List

from pywrapr.docs_conversion import translate_python_row

def is_import_line(line: str):
    return line.startswith('import ') or ' import ' in line

def transpile_py_to_r(lines: List[str]):
    # must_load_library = any(map(is_import_line, lines))
    filtered_lines = [line for line in lines if not(is_import_line(line))]

    transpiled_lines = [translate_python_row(line) for line in filtered_lines]

    return transpiled_lines
    # return ['library(pharmr)', *transpiled_lines] if must_load_library else list(transpiled_lines)

