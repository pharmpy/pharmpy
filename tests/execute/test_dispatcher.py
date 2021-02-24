from pharmpy.execute.databases.local_directory import LocalDirectoryDatabase
from pharmpy.execute.dispatchers.local import LocalDispatcher
from pharmpy.methods.modelfit.job import ModelfitJob


def fun(s):
    return s


def test_local_dispatcher():
    db = LocalDirectoryDatabase()
    disp = LocalDispatcher()
    wfl = {'results': (fun, 'input')}
    job = ModelfitJob(wfl)
    res = disp.run(job, db)
    assert res == 'input'
