import warnings

import numpy as np

from pharmpy.modeling import update_inits


def update_initial_estimates(model):
    try:
        update_inits(model, move_est_close_to_bounds=True)
    except (ValueError, np.linalg.LinAlgError):
        warnings.warn(f'{model.name}: Could not update initial estimates, using original estimates')
        pass
    return model
