from pathlib import Path
import re
import sys

# This script will check that all exported functions in modeling and tools have
# typehints for all arguments. It can be replaced with ANN001 in ruff
# It doesn't use introspection to not need pharmpy installed and to be fast


def remove_nested_brackets(text: str, start: str, end: str) -> str:
    result = []
    depth = 0

    for char in text:
        if char == start:
            depth += 1
        elif char == end:
            if depth > 0:
                depth -= 1
            else:
                result.append(char)
        elif depth == 0:
            result.append(char)

    return "".join(result)


def remove_quotation(text : str) -> str:
    result = []
    inquote = None
    for char in text:
        if not inquote:
            if char in {'"', "'"}:
                inquote = char
            else:
                result.append(char)
        elif char == inquote:
                inquote = None

    return "".join(result)


def check_type_annotations(files, funcs):
    global error
    for file in files:
        with open(file, "r") as f:
            curfunc = None
            for line in f:
                line = line.strip()
                if curfunc is None and line.startswith("def "):
                    name, _, curline = line[4:].partition("(")
                    if name in funcs:
                        curfunc = name
                        parennest = 1 + curline.count("(") - curline.count(")")
                elif curfunc is not None:
                    curline += line
                    parennest += line.count("(") - line.count(")")
                if curfunc is not None and parennest == 0:
                    args = remove_nested_brackets(curline, start="[", end="]")
                    args = remove_quotation(args)
                    args = remove_nested_brackets(args, start="(", end=")")

                    args, _, result = args.partition(")")
                    args = args.split(',')
                    all_args = True
                    for arg in args:
                        if not arg:
                            continue
                        if ':' not in arg and 'kwargs' not in arg:
                            all_args = False
                            break
                    if not all_args:
                        print(f"Missing type annotations for {curfunc} in {file}")
                        error = True
                    curfunc = None


def strip_comment(line):
    code, _, _ = line.partition("#")
    return code


def parse_funcs_from_all(path):
    names = set()
    with open(path, "r") as f:
        in_all = False
        for line in f:
            if line.startswith('__all__'):
                in_all = True
            elif in_all:
                line = strip_comment(line)
                if ')' in line or ']' in line:
                    in_all = False
                else:
                    line = line.strip()[1:]
                    name, _, _ = line.partition("'")  # Assumes ' around func names
                    names.add(name)
    return names


path = Path(sys.argv[1])

error = False

funcs = parse_funcs_from_all(path / "modeling" / "__init__.py")
files = list((path / "modeling").glob("*.py"))
check_type_annotations(files, funcs)

funcs = parse_funcs_from_all(path / "tools" / "__init__.py")
funcs = {func for func in funcs if not func.startswith("run_")} | {"create_workflow"}
files = list((path / "tools").glob("**/*.py"))
check_type_annotations(files, funcs)

if error:
    raise SyntaxError("Some missing")
