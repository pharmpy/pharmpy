from dask.multiprocessing import get

import pharmpy.plugins.nonmem.run as nmrun


def run(models, path):
    dsk = nmrun.run(models, path)
    res = get(dsk, 'results')  # executes in parallel
    return res
