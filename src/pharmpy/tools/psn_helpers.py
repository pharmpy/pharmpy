import re
import shutil
import sys
import zipfile
from pathlib import Path


def options_from_command(command):
    p = re.compile('^-+([^=]+)=?(.*)')
    return {
        match.group(1): match.group(2)
        for val in command.split()
        if val.startswith('-') and (match := p.match(val))
    }


def arguments_from_command(command):
    command = re.sub(r'command_line:', '', command)
    return [arg for arg in command.split()[1:] if not arg.startswith('-')]


def model_paths(path, pattern, subpath='m1'):
    path = Path(path) / subpath
    model_paths = list(path.glob(pattern))
    model_paths.sort(key=lambda name: int(re.sub(r'\D', '', str(name))))
    return model_paths


def tool_from_command(command):
    command = re.sub(r'command_line:', '', command)
    tool_with_path = command.split(maxsplit=1)[0]
    return Path(tool_with_path).name


def psn_directory_list(path, drop_tools=[]):
    path = Path(path)
    folder_list = [{'name': p.name, 'tool': tool_name(p)} for p in path.iterdir() if p.is_dir()]
    return [d for d in folder_list if d['tool'] is not None and not d['tool'] in drop_tools]


def tool_name(path):
    command = psn_command(path)
    if command is not None:
        return tool_from_command(command)


def psn_command(path):
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        with open(path / 'command.txt') as meta:
            return next(meta)
    except FileNotFoundError:
        pass
    try:
        with open(path / 'meta.yaml') as meta:
            cmd = None
            for row in meta:
                if row.startswith('command_line: '):
                    cmd = row.strip()
                elif cmd is not None:
                    # command can be split over multiple lines
                    if row.startswith('common_options:'):
                        return cmd
                    elif re.match(r'\s', row):  # continuation is indented
                        cmd += row.strip()
            if cmd is not None:
                return cmd
    except FileNotFoundError:
        pass


def cmd_line_model_path(path):
    path = Path(path)
    with open(path / 'meta.yaml') as meta:
        for row in meta:
            row = row.strip()
            if row.startswith('model_files:'):
                row = next(meta).strip()
                return Path(re.sub(r'^-\s*', '', row))


def template_model_string(datafile=None, ignore=[], drop=[], label='TEMPLATE'):
    variables = ''
    indent = r'      '
    ignores = indent + 'IGNORE=@'
    if len(ignore) > 0:
        ignores += '\n' + '\n'.join([indent + f'IGNORE=({ign})' for ign in ignore])
    if datafile is None:
        datafile = 'dummydata.csv'
    else:
        datafile = Path(datafile)
        try:
            with open(datafile) as data:
                row = next(data)
                row = row.strip()
                if re.match(r'[A-Z]', row):
                    names = row.split(r',')
                    variables = ' '
                    start = 0
                    while start < len(names):
                        stop = min(start + 10, len(names))
                        variables += (
                            ' '.join([n if n not in drop else 'DROP' for n in names[start:stop]])
                            + '\n      '
                        )
                        start = stop
                else:
                    pass  # unsupported header type
        except FileNotFoundError:
            pass
    return '\n'.join(
        [
            '$PROBLEM ' + label,
            '$INPUT' + variables,
            f'$DATA {str(datafile)}',
            ignores,
            '$SUBROUTINE ADVAN1 TRANS2',
            '',
            '$PK',
            'CL=THETA(1)*EXP(ETA(1))',
            'V=THETA(2)*EXP(ETA(2))',
            'S1=V',
            '',
            '$ERROR',
            'Y=F+F*EPS(1)',
            '',
            '$THETA (0, 1)       ; TVCL',
            '$THETA (0, 5)       ; TVV',
            '$OMEGA 0.1           ; IVCL',
            '$OMEGA 0.1           ; IVV',
            '$SIGMA 0.025         ; RUV',
            '',
            '$ESTIMATION METHOD=1 INTERACTION',
            '$COVARIANCE PRINT=E MATRIX=S UNCONDITIONAL',
        ]
    )


def pharmpy_wrapper():
    """Command line wrapper for PsN to call pharmpy"""
    exec(sys.argv[1], globals(), {})


def create_results(path, **kwargs):
    path = Path(path)
    name = tool_name(path)
    m1zip = path / "m1.zip"
    m1 = path / "m1"
    if m1zip.is_file() and not m1.is_dir():
        unzipped = True
        with zipfile.ZipFile(m1zip, 'r') as zip_ref:
            zip_ref.extractall(path)
    else:
        unzipped = False

    # FIXME: Do something automatic here
    if name == 'qa':
        from pharmpy.tools.qa.results import psn_qa_results

        res = psn_qa_results(path, **kwargs)
    elif name == 'bootstrap':
        from pharmpy.tools.bootstrap.results import psn_bootstrap_results

        res = psn_bootstrap_results(path, **kwargs)
    elif name == 'cdd':
        from pharmpy.tools.cdd.results import psn_cdd_results

        res = psn_cdd_results(path, **kwargs)
    elif name == 'frem':
        from pharmpy.tools.frem.results import psn_frem_results

        res = psn_frem_results(path, **kwargs)
    elif name == 'linearize':
        from pharmpy.tools.linearize.results import psn_linearize_results

        res = psn_linearize_results(path, **kwargs)
    elif name == 'ruvsearch':
        from pharmpy.tools.ruvsearch.results import psn_resmod_results

        res = psn_resmod_results(path, **kwargs)

    elif name == 'scm':
        from pharmpy.tools.scm.results import psn_scm_results

        res = psn_scm_results(path, **kwargs)
    elif name == 'simeval':
        from pharmpy.tools.simeval.results import psn_simeval_results

        res = psn_simeval_results(path, **kwargs)
    elif name == 'crossval':
        from pharmpy.tools.crossval.results import psn_crossval_results

        res = psn_crossval_results(path, **kwargs)
    else:
        raise ValueError("Not a valid run directory")

    if unzipped:
        shutil.rmtree(path / 'm1')

    return res
