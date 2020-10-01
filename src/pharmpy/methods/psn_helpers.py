import re
from pathlib import Path


def options_from_command(command):
    p = re.compile('^-+([^=]+)=?(.*)')
    return {p.match(val).group(1): p.match(val).group(2)
            for val in command.split() if val.startswith('-')}


def arguments_from_command(command):
    command = re.sub(r'command_line:', '', command)
    return [arg for arg in command.split()[1:] if not arg.startswith('-')]


def model_paths(path, pattern):
    path = Path(path) / 'm1'
    model_paths = list(path.glob(pattern))
    model_paths.sort(key=lambda name: int(re.sub(r'\D', '', str(name))))
    return model_paths


def tool_from_command(command):
    command = re.sub(r'command_line:', '', command)
    tool_with_path = command.split(maxsplit=1)[0]
    return Path(tool_with_path).name


def psn_directory_list(path, drop_tools=list()):
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
