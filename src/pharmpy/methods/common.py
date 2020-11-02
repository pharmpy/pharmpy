from pathlib import Path


class RunDirectory:
    def __init__(self, method_name, path=None):
        i = 1
        while True:
            name = f'{method_name}_dir{i}'
            if path is not None:
                path = path / name
            else:
                path = Path(name)
            if not path.is_file():
                path.mkdir()
                self.path = path
                self.models_path = path / 'models'
                self.models_path.mkdir()
                break
            i += 1


class Method:
    def __init__(self, path=None):
        self.rundir = RunDirectory(type(self).__name__.lower(), path=path)
