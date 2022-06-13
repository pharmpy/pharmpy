from .run import create_results, fit, read_results, run_tool
from .run_amd import run_amd
from .wrap import wrap

run_allometry = wrap('allometry')
run_iivsearch = wrap('iivsearch')
run_modelsearch = wrap('modelsearch')
run_resmod = wrap('resmod')

__all__ = [
    'create_results',
    'fit',
    'read_results',
    'run_allometry',
    'run_amd',
    'run_iivsearch',
    'run_modelsearch',
    'run_resmod',
    'run_tool',
]
