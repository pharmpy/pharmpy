# Functional interface to extract dataset information


def ninds(model):
    return len(model.dataset.pharmpy.ids)


def nobs(model):
    return len(model.dataset.pharmpy.observations)
