# All functions used for comparing a set of candidate models and
# ranking them take the same base arguments
# base - the base model
# candidates - a list of candidate models
# returns a list which is a subset of the candidates with the best ranked at the beginning
# models can be removed from candidates if they don't pass some criteria


def ofv(base, candidates, cutoff=3.84):
    base_ofv = base.modelfit_results.ofv
    filtered = [model for model in candidates if base_ofv - model.modelfit_results.ofv >= cutoff]
    srtd = sorted(filtered, key=lambda model: model.modelfit_results.ofv)
    return srtd
