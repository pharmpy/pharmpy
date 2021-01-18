from pathlib import Path


class RunDirectory:
    def __init__(self, method_name, path=None):
        i = 1
        while True:
            name = f'{method_name}_dir{i}'
            if path is not None:
                test_path = path / name
            else:
                test_path = Path(name)
            if not test_path.exists():
                test_path.mkdir()
                self.path = test_path
                self.models_path = test_path / 'models'
                self.models_path.mkdir()
                break
            i += 1


class Method:
    def __init__(self, path=None):
        self.rundir = RunDirectory(type(self).__name__.lower(), path=path)
