import re
from pathlib import Path


def options_from_command(command):
    p = re.compile('^-+([^=]+)=?(.*)')
    return {p.match(val).group(1): p.match(val).group(2)
            for val in command.split() if val.startswith('-')}


def model_paths(path, pattern):
    path = Path(path) / 'm1'
    model_paths = list(path.glob(pattern))
    model_paths.sort(key=lambda name: int(re.sub(r'\D', '', str(name))))
    return model_paths


def cmd_line_model_path(path):
    path = Path(path)
    with open(path / 'meta.yaml') as meta:
        for row in meta:
            row = row.strip()
            if row.startswith('model_files:'):
                row = next(meta).strip()
                return Path(re.sub(r'^-\s*', '', row))
