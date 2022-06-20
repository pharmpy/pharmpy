from .run import create_results, fit, read_results, retrieve_models, run_tool
from .run_amd import run_amd
from .wrap import wrap

run_allometry = wrap('allometry')
run_covsearch = wrap('covsearch')
run_iivsearch = wrap('iivsearch')
run_iovsearch = wrap('iovsearch')
run_modelsearch = wrap('modelsearch')
run_resmod = wrap('resmod')

__all__ = [
    'create_results',
    'fit',
    'read_results',
    'retrieve_models',
    'run_allometry',
    'run_amd',
    'run_covsearch',
    'run_iivsearch',
    'run_iovsearch',
    'run_modelsearch',
    'run_resmod',
    'run_tool',
]
