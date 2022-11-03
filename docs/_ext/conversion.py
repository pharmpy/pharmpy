import re
from inspect import getdoc
from typing import List

def create_r_doc(func):
    doc = getdoc(func)

    if not doc:
        return f'#\' @title\n' \
               f'#\' {func.__name__}\n' \
               f'#\' \n' \
               f'#\' @export'

    doc_dict = split_doc_to_subtypes(doc)

    doc_str = f'@title\n{func.__name__}\n\n@description\n'

    for row in doc_dict['description']:
        is_list = row.startswith('-')
        if is_list:
            row = re.sub('^- ', '* ', row)
        if row.startswith('.. '):
            row = re.sub(r'^\.\. ', '', row)
        doc_str += f'{py_to_r_str(row)}\n'

    if 'params' in doc_dict.keys():
        doc_str += create_r_params(doc_dict['params'])
    if 'returns' in doc_dict.keys():
        doc_str += create_r_returns(doc_dict['returns']) + '\n\n'
    if 'examples' in doc_dict.keys():
        doc_str += create_r_example(doc_dict['examples']) + '\n'
    if 'notes' in doc_dict.keys():
        doc_str += '@note\n'
        doc_str += ''.join(doc_dict['notes']) + '\n\n'
    if 'see_also' in doc_dict.keys():
        doc_str += '@seealso\n'
        doc_str += '\n\n'.join(doc_dict['see_also']) + '\n\n'

    r_doc = ''
    for row in doc_str.split('\n'):
        r_doc += f'#\' {row}\n'

    r_doc += f'#\' @export'
    return r_doc


def create_r_params(doc_list):
    doc_str = ''
    skip = False
    for i, row in enumerate(doc_list):
        if skip:
            skip = False
            continue
        type_declare_pattern = re.compile(r'([\w0-9]+) : ([\w0-9]+)')
        if type_declare_pattern.match(row):
            type_declare = row.split(' : ')
            doc_str += f'@param {type_declare[0]} ({py_to_r_str(type_declare[1])})'
        elif row == 'args' or row == 'kwargs':
            doc_str += f'@param ... {doc_list[i+1]}\n'
            skip = True
        else:
            doc_str += f' {py_to_r_str(row)}\n'
    return doc_str


def create_r_returns(doc_list):
    doc_list_trim = list(filter(None, doc_list))
    type_py = doc_list_trim[0]
    type_r = py_to_r_str(type_py)

    doc_str = f'@return ({type_r}) '
    doc_str += ' '.join(py_to_r_str(row) for row in doc_list_trim[1:])

    return doc_str

pattern_start = re.compile(r'>>> |^\.\.\. ')
pattern_methods = re.compile(r'([A-Za-z_]\d*)\.([A-Za-z_]\d*)(?=(?:[^"]|"[^"]*")*$)(?=(?:[^\']|\'[^\']*\')*$)')
pattern_list_idx = re.compile(r'\w\[(\d+)]')
pattern_list = re.compile(r'\[([\'\"\w(),\s]+)](?=(?:[^"]|"[^"]*")*$)(?=(?:[^\']|\'[^\']*\')*$)')
pattern_dict = re.compile(r'{(([\'\"]*[\w\d()]+[\'\"]*: [\'\"]*[\w\d]+\'*,*\s*)+)}')
pattern_doctest = re.compile(r'\s+# doctest:.*')

def line_transpile_py_to_r(line: str) -> str:
    line = py_to_r_str(line, example=True)
    line = re.sub(' = ', ' <- ', line)

    # Substitute any . to $, e.g. model.parameters -> model$parameters
    line = re.sub(pattern_methods, r'\1$\2', line)

    # Substitute """ or ''' to " for multiline strings
    line = re.sub(r'"""', r'"', line)
    line = re.sub(r"'''", r'"', line)

    # Check if line has list subscript
    if re.search(r'\w\[', line):
        idx = re.search(pattern_list_idx, line)
        if idx:
            # Increase idx by 1, e.g. list[0] -> list[1]
            line = re.sub(idx.group(1), f'{int(idx.group(1)) + 1}', line)
    else:
        # Substitute any [] to c(), e.g. ['THETA(1)'] -> c('THETA(1)')
        line = re.sub(pattern_list, r'c(\1)', line)

    # Check if line contains python dict
    dict_py = re.search(pattern_dict, line)
    if dict_py:
        # Substitute {} to list(), e.g. {'EONLY': 1} -> list('EONLY'=1)
        dict_r = f'list({dict_py.group(1).replace(": ", "=", )})'
        line = line.replace('{' + f'{dict_py.group(1)}' + '}', dict_r)

    # Replace len() with length()
    line = line.replace(r'len(', 'length(')

    # Remove doctest comments
    line = re.sub(pattern_doctest, '', line)
    return line

def is_import_line(line: str):
    return line.startswith('import ') or ' import ' in line

def transpile_py_to_r(lines: List[str]):
    # must_load_library = any(map(is_import_line, lines))
    filtered_lines = [line for line in lines if not(is_import_line(line))]

    transpiled_lines = map(
        lambda line: line_transpile_py_to_r(re.sub(pattern_start, '', line)),
        filtered_lines,
    )

    return transpiled_lines
    # return ['library(pharmr)', *transpiled_lines] if must_load_library else list(transpiled_lines)

def create_r_example(doc_list):
    # Check for rows that starts with ... or >>>
    doc_code = [row for row in doc_list if row.startswith('>>>') or re.match(r'^\.\.\.\s+[$\w\d]', row)]

    doc_code_r = map(
        lambda line: line  + '\n',
        transpile_py_to_r([
            re.sub(pattern_start, '', line) for line in doc_code
        ])
    )

    return '@examples\n\\dontrun{\n' + ''.join(doc_code_r) + '}'


def split_doc_to_subtypes(doc_str):
    doc_split = doc_str.split('\n')

    doc_titles = {'Parameters': 'params',
                  'Returns': 'returns',
                  'Return': 'returns',
                  'Results': 'returns',
                  'Example': 'examples',
                  'Examples': 'examples',
                  'Notes': 'notes',
                  'See also': 'see_also',
                  'See Also': 'see_also'}

    doc_type_current = 'description'

    doc_dict = {}
    doc_dict[doc_type_current] = []

    for row in doc_split:
        if row in doc_titles.keys() and doc_titles[row] not in doc_dict.keys():
            doc_type_current = doc_titles[row]
            doc_dict[doc_type_current] = []
            continue

        if not row.startswith('--'):
            doc_dict[doc_type_current].append(row.strip())

    return doc_dict


def py_to_r_arg(arg):
    py_to_r_dict = {'None': 'NULL',
                    'True': 'TRUE',
                    'False': 'FALSE',
                    '': '\'\''}

    try:
        return py_to_r_dict[str(arg)]
    except KeyError:
        if isinstance(arg, str):
            return f'\'{arg}\''
        else:
            return arg


def py_to_r_str(arg, example=False):
    args = {'None': 'NULL',
            'True': 'TRUE',
            'False': 'FALSE'}

    types = {r'\bint\b': 'integer',
             'float': 'numeric',
             r'\bbool\b': 'logical',
             r'\blist\b': 'vector',
             r'\bdict\b': 'list',
             'dictionary': 'list',
             'pd.DataFrame': 'data.frame',
             'pd.Series': 'data.frame',
             r'List\[Model\]': 'vector of Models'}  # FIXME: more general pattern

    latex = {r'\\mathsf': '',
             r'\\cdot': '*',
             r'\\text': '',
             r'\\frac': 'frac',
             r'\\log': 'log',
             r'\\exp': 'exp',
             r'\\min': 'min',
             r'\\max': 'max',
             r'\\epsilon': 'epsilon'}

    py_to_r_dict = {**args, **types, **latex}

    if not example:
        py_to_r_dict = {**py_to_r_dict, r'\[([0-9]+)\]_*': r'(\1)'}

    arg_sub = arg
    for key, value in py_to_r_dict.items():
        arg_sub = re.sub(key, value, arg_sub)

    return arg_sub

