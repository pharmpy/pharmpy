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
