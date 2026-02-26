from dataclasses import dataclass

from pharmpy.basic import Quantity, Unit
from pharmpy.deps import pandas as pd
from pharmpy.deps.rich import box as rich_box
from pharmpy.deps.rich import console as rich_console
from pharmpy.deps.rich import table as rich_table
from pharmpy.model import DataVariable, Model


def get_variable_description(model, variable) -> str:
    for col in model.datainfo:
        if col.variable_id is None:
            if variable == col.variable_mapping:
                return col.name
        elif variable in col.variable_mapping.values():
            inv_mapping = {v: k for k, v in col.variable_mapping.items()}
            i = inv_mapping[variable]
            return f"{col.name}[{col.variable_id} == {i}]"


def get_variable_data(model, variable) -> pd.Series:
    for col in model.datainfo:
        if col.variable_id is None:
            if variable == col.variable_mapping:
                return model.dataset[col.name]
        elif variable in col.variable_mapping.values():
            inv_mapping = {v: k for k, v in col.variable_mapping.items()}
            i = inv_mapping[variable]
            return model.dataset.loc[model.dataset[col.variable_id] == i, col.name]


class Violation:
    pass


@dataclass
class DatasetViolation(Violation):
    model: Model
    variable: DataVariable
    rows: list[int]
    msg: str

    def __str__(self):
        return f"{self.msg} in {get_variable_description(self.model, self.variable)} at {self.rows}"


@dataclass
class DataInfoViolation(Violation):
    model: Model
    variable: DataVariable
    msg: str

    def __str__(self):
        return f"{self.msg} for {get_variable_description(self.model, self.variable)}"


class VariableQuantifier:
    pass


@dataclass
class Descriptor(VariableQuantifier):
    value: str

    def get_variables(self, model: Model) -> list[DataVariable]:
        di = model.datainfo
        variables = [
            variable
            for variable in di.variables
            if variable.properties.get("descriptor", None) == self.value
        ]
        return variables

    def __str__(self):
        return f'All data variables with descriptor="{self.value}"'


@dataclass
class Type(VariableQuantifier):
    value: str

    def get_variables(self, model: Model) -> list[DataVariable]:
        di = model.datainfo
        try:
            variables = di.typeix[self.value].variables
        except IndexError:
            variables = []
        return variables

    def __str__(self):
        return f"All {self.value} columns"


class Predictor:
    pass


@dataclass
class UnitOf(Predictor):
    variables: VariableQuantifier
    unit: Unit

    def check(self, model) -> list[Violation]:
        violations = []
        for variable in self.variables.get_variables(model):
            diunit = variable.get_property("unit")
            if not diunit.is_compatible_with(self.unit):
                violation = DataInfoViolation(model, variable, "Unit not compatible")
                violations.append(violation)
        return violations

    def __str__(self):
        if self.unit == Unit.unitless():
            return f"{self.variables} are dimensionless"
        else:
            return f"{self.variables} have dimension {self.unit.get_dimensionality_string()}"


@dataclass
class InRange(Predictor):
    variables: VariableQuantifier
    unit: Unit
    lower: float
    upper: float = float("inf")
    lower_included: bool = True
    upper_included: bool = False

    def check(self, model) -> list[Violation]:
        violations = []
        for variable in self.variables.get_variables(model):
            variable_unit = variable.get_property("unit")
            if not variable_unit.is_compatible_with(self.unit):
                continue
            violated_indices = []
            ser = get_variable_data(model, variable)
            converted_lower = Quantity(self.lower, self.unit).convert_to(variable_unit).value
            converted_upper = Quantity(self.upper, self.unit).convert_to(variable_unit).value
            for i, val in ser.items():
                if (
                    self.lower_included
                    and val < converted_lower
                    or not self.lower_included
                    and val <= converted_lower
                ):
                    violated_indices.append(i)
                elif (
                    self.upper_included
                    and val > converted_upper
                    or not self.upper_included
                    and val >= converted_upper
                ):
                    violated_indices.append(i)
            if violated_indices:
                violation = DatasetViolation(
                    model, variable, violated_indices, "Value out of range"
                )
                violations.append(violation)
        return violations

    def __str__(self):
        lower_op = "≤" if self.lower_included else "<"
        upper_op = "≤" if self.upper_included else "<"
        upper_part = f" {upper_op} {self.upper}" if self.upper < float("inf") else ""
        return f"{self.variables} are in the range {self.lower} {lower_op} x{upper_part}"


@dataclass
class InSet(Predictor):
    variables: VariableQuantifier
    values: set[float | int]

    def check(self, model) -> list[Violation]:
        violations = []
        for variable in self.variables.get_variables(model):
            violated_indices = []
            ser = get_variable_data(model, variable)
            for i, val in ser.items():
                if val not in self.values:
                    violated_indices.append(i)
            if violated_indices:
                violation = DatasetViolation(
                    model, variable, violated_indices, "Value not in allowed set"
                )
                violations.append(violation)
        return violations

    def __str__(self):
        return f"{self.variables} ∈ {self.values}"


PREDICATES = (
    UnitOf(Type("id"), Unit(1)),
    InRange(Type("id"), Unit(1), 1),
    UnitOf(Type("dvid"), Unit(1)),
    InRange(Type("dvid"), Unit(1), 0),
    UnitOf(Type("mdv"), Unit(1)),
    InSet(Type("mdv"), {0, 1}),
    UnitOf(Descriptor("body weight"), Unit("kg")),
    InRange(Descriptor("body weight"), Unit("kg"), 0.0, 700.0, lower_included=False),
    UnitOf(Descriptor("lean body mass"), Unit("kg")),
    InRange(Descriptor("lean body mass"), Unit("kg"), 0.0, 700.0, lower_included=False),
    UnitOf(Descriptor("fat free mass"), Unit("kg")),
    InRange(Descriptor("fat free mass"), Unit("kg"), 0.0, 700.0, lower_included=False),
    UnitOf(Descriptor("age"), Unit("yr")),
    InRange(Descriptor("age"), Unit("yr"), 0.0, 130.0),
    UnitOf(Descriptor("time after dose"), Unit("h")),
    InRange(Descriptor("time after dose"), Unit("h"), 0.0),
    UnitOf(Descriptor("plasma concentration"), Unit("mg/L")),
    InRange(Descriptor("plasma concentration"), Unit("h"), 0.0),
)


def pretty_print_checks(checks):
    table = rich_table.Table(title="Dataset checks", box=rich_box.SQUARE, show_lines=True)
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Violations")

    for check, result, violation in checks:
        if result == "OK":
            table.add_row(check, f'[bold green]{result}', "")
        else:
            table.add_row(check, f'[bold red]{result}', violation)

    if table.rows:  # Do not print an empty table
        console = rich_console.Console()
        console.print(table)


def check_dataset(model: Model, dataframe: bool = False, verbose: bool = False):
    """Check dataset for consistency across a set of rules

    Parameters
    ----------
    model : Model
        Pharmpy model
    dataframe : bool
        True to return a DataFrame instead of printing to the console
    verbose : bool
        Print out all rules checked if True else print only failed rules

    Returns
    -------
    pd.DataFrame
        Only returns a DataFrame is dataframe=True
    """

    checks = []
    for pred in PREDICATES:
        violations = pred.check(model)
        rule_string = str(pred)
        if not violations:
            if verbose:
                checks.append((rule_string, "OK", ""))
        else:
            for violation in violations:
                checks.append((rule_string, "FAIL", str(violation)))

    if not dataframe:
        pretty_print_checks(checks)
    else:
        df = pd.DataFrame(checks, columns=['check', 'result', 'violations'])
        return df
