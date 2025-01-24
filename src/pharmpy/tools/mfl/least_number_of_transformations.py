import warnings
from typing import Optional

from pharmpy.model import Model
from pharmpy.tools.mfl.feature.covariate import features as covariate_features
from pharmpy.tools.mfl.parse import ModelFeatures, _add_helper
from pharmpy.tools.mfl.statement.feature.covariate import Covariate


def least_number_of_transformations(
    self, other, model: Optional[Model] = None, tool: Optional[str] = None
):
    """The smallest set of transformations to become part of other"""

    def _lnt_helper(lhs, rhs, mfl, name, lnt):
        if lhs is None and rhs is not None or lhs is not None and rhs is None:
            raise ValueError(
                f"{name} : is only part of one of the MFLs" " and therefore cannot be compared"
            )
        if lhs is None and rhs is None:
            return lnt
        if not any(x in rhs.eval.modes for x in lhs.eval.modes):
            name, func = list(mfl.convert_to_funcs([name]).items())[0]
            lnt[name] = func
        return lnt

    # Add more tools than "modelsearch" if support is needed
    lnt = {}
    if tool is None or tool in ["modelsearch"]:
        lnt = _lnt_helper(self.absorption, other.absorption, other, "absorption", lnt)
        lnt = _lnt_helper(self.elimination, other.elimination, other, "elimination", lnt)
        lnt = _lnt_transits(self, other, lnt)
        lnt = _lnt_peripherals(self, other, lnt, "pk")
        lnt = _lnt_helper(self.lagtime, other.lagtime, other, "lagtime", lnt)

    # TODO : Use in covsearch instead of taking diff
    if tool is None:
        if model is not None:
            lnt = _lnt_covariates(self, other, lnt, model)
        else:
            if self.covariate != tuple() or other.covariate != tuple():
                warnings.warn("Need argument 'model' in order to compare covariates")

        lnt = _lnt_peripherals(self, other, lnt, "metabolite")
        lnt = _lnt_helper(self.direct_effect, other.direct_effect, other, "direct_effect", lnt)
        lnt = _lnt_helper(self.effect_comp, other.effect_comp, other, "effect_comp", lnt)
        lnt = _lnt_indirect_effect(self, other, lnt)
        lnt = _lnt_helper(self.metabolite, other.metabolite, other, "metabolite", lnt)

    return lnt


def _lnt_indirect_effect(self, other, lnt):
    lhs, rhs, combine = _add_helper(
        self.indirect_effect, other.indirect_effect, "modes", "production"
    )
    if not combine and rhs:
        # No shared attribute
        func_dict = other.convert_to_funcs(["indirect_effect"])
        for key in lhs.keys():
            if key in rhs.keys():
                lnt[('INDIRECT', rhs[key][0], key.name)] = func_dict[
                    ('INDIRECT', rhs[key][0], key.name)
                ]
                return lnt
        # No key is matching
        key = next(iter(rhs))
        lnt[('INDIRECT', rhs[key][0], key.name)] = func_dict[('INDIRECT', rhs[key][0], key.name)]
        return lnt
    return lnt


def _lnt_transits(self, other, lnt):
    lhs, rhs, combine = _add_helper(self.transits, other.transits, "counts", "depot")
    if not combine and rhs:
        # No shared attribute
        func_dict = other.convert_to_funcs(["transits"])
        for key in lhs.keys():
            if key in rhs.keys():
                lnt[('TRANSITS', rhs[key][0], key.name)] = func_dict[
                    ('TRANSITS', rhs[key][0], key.name)
                ]
                return lnt
        # No key is matching
        key = next(iter(rhs))
        lnt[('TRANSITS', rhs[key][0], key.name)] = func_dict[('TRANSITS', rhs[key][0], key.name)]
        return lnt
    return lnt


def _lnt_peripherals(self, other, lnt, subset):
    if subset == "pk":
        keys = ["DRUG"]
    elif subset == "metabolite":
        keys = ["MET"]
    else:
        keys = ["DRUG", "MET"]
    lhs = self._extract_peripherals()
    rhs = other._extract_peripherals()
    func_dict = other.convert_to_funcs(["peripherals"])
    for key in keys:
        if not any(c in rhs[key] for c in lhs[key]):
            if key == "DRUG":
                if rhs[key]:
                    lnt[("PERIPHERALS", min(rhs[key]))] = func_dict[("PERIPHERALS", min(rhs[key]))]
            elif key == "MET":
                if rhs[key]:
                    lnt[("PERIPHERALS", min(rhs[key]), "METABOLITE")] = func_dict[
                        ("PERIPHERALS", min(rhs[key]), "METABOLITE")
                    ]
    return lnt


def _lnt_covariates(self, other, lnt, model):
    lhs = self._extract_covariates()
    lhs = [c for c in lhs if not c[4].option]  # Check only FORCED
    rhs = other._extract_covariates()
    rhs = [c for c in rhs if not c[4].option]  # Check only FORCED

    def convert_to_covariate(combinations):
        cov_list = []
        for param, cov, fp, op, opt in combinations:
            cov_list.append(Covariate((param,), (cov,), (fp,), op, opt))
        return cov_list

    # Remove all unqiue to LHS
    lhs_unique = [c for c in lhs if c not in rhs]
    if lhs_unique:
        cov_list = convert_to_covariate(lhs_unique)
        lhs_mfl = ModelFeatures.create_from_mfl_statement_list(cov_list)
        remove_cov_dict = dict(covariate_features(model, lhs_mfl.covariate, remove=True))
        lnt.update(remove_cov_dict)

    # Add all unique to RHS
    rhs_unique = [c for c in rhs if c not in lhs]
    if rhs_unique:
        cov_list = convert_to_covariate(rhs_unique)
        lhs_mfl = ModelFeatures.create_from_mfl_statement_list(cov_list)
        add_cov_dict = dict(covariate_features(model, lhs_mfl.covariate, remove=False))
        lnt.update(add_cov_dict)

    return lnt
