from pathlib import Path

import pharmpy.execute as execute


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
    def __init__(self, dispatcher=None, database=None, job_creator=None, path=None):
        self.rundir = RunDirectory(type(self).__name__.lower(), path=path)
        if dispatcher is None:
            self.dispatcher = execute.default_dispatcher
        else:
            self.dispatcher = dispatcher
        if database is None:
            self.database = execute.default_database
        else:
            self.database = database
        if job_creator is None:
            import pharmpy.plugins.nonmem.run

            self.job_creator = pharmpy.plugins.nonmem.run.create_job
        else:
            self.job_creator = job_creator
